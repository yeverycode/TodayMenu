from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from passlib.context import CryptContext
import pandas as pd
import torch
import torch.nn as nn

# 내부 모듈 Import
from mypage_api import mypage_router
from backend.ai_recommend_api import router as rule_recommend_router
from review_api import router as review_router
from ai.llm_service import router as llm_router
from backend.register import router as register_router
from backend.llm_recommend_api import router as llm_recommend_router  # Streaming LLM용
from models import User, SessionLocal
from ai.langchain_recommender import recommend_menu as llm_recommend_menu
from feedback_api import router as feedback_router 

app = FastAPI(title="오늘의 먹방은 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 라우터 등록 (prefix 주의)
app.include_router(register_router, tags=["auth"])
app.include_router(mypage_router, tags=["mypage"])
app.include_router(review_router, prefix="/api", tags=["review"])  # 수정됨
app.include_router(rule_recommend_router, prefix="/api", tags=["rule-recommend"])
app.include_router(llm_router, tags=["llm"])
app.include_router(llm_recommend_router, prefix="/api", tags=["llm-streaming"])
app.include_router(feedback_router, prefix="/api", tags=["feedback"]) 

# 비밀번호 암호화 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------
# 회원가입 API
# -------------------------------
class UserCreate(BaseModel):
    username: str
    password: str
    allergies: str

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = pwd_context.hash(user.password)
    new_user = User(username=user.username, password=hashed_pw, allergies=user.allergies)
    db.add(new_user)
    db.commit()

    return {"msg": "User registered"}

# -------------------------------
# 로그인 API
# -------------------------------
class UserLogin(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not pwd_context.verify(user.password, db_user.password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    return {"username": db_user.username, "allergies": db_user.allergies}

# -------------------------------
# Rule 기반 추천 API
# -------------------------------
class RecommendRequest(BaseModel):
    username: str
    weather: str

menu_df = pd.read_csv("data/menu_price.csv")
nutrient_df = pd.read_csv("data/menu_nutrient.csv")

@app.post("/api/recommend")
def recommend(request: RecommendRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    allergies = user.allergies.split(",") if user.allergies else []
    merged = menu_df.merge(nutrient_df, left_on="menu_name", right_on="name", how="left")

    def has_allergy(allergy_str, allergies):
        if pd.isna(allergy_str):
            return False
        return any(allergy in allergy_str for allergy in allergies)

    filtered = merged[~merged["allergy"].apply(lambda x: has_allergy(x, allergies))]
    filtered = filtered[filtered["weather"].str.contains(request.weather, na=False)]

    result = filtered[["place_name", "menu_name", "menu_price"]].to_dict(orient="records")
    return {"recommendations": result}

# -------------------------------
# AI 기반 추천 (PyTorch)
# -------------------------------
class RecommendInput(BaseModel):
    user_data: list

INPUT_DIM = 20
OUTPUT_DIM = 50

class MenuRecommender(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MenuRecommender, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

model = MenuRecommender(INPUT_DIM, 64, OUTPUT_DIM)
model.load_state_dict(torch.load("menu_model.pth"))
model.eval()

menu_list = [f"메뉴{i}" for i in range(OUTPUT_DIM)]

@app.post("/api/ai_recommend")
def ai_recommend(input_data: RecommendInput):
    user_tensor = torch.tensor([input_data.user_data], dtype=torch.float32)

    with torch.no_grad():
        prediction = model(user_tensor)
        top_menu_index = torch.argmax(prediction).item()

    recommended_menu = menu_list[top_menu_index]
    return {"recommended_menu": recommended_menu}

# -------------------------------
# LLM LangChain 기반 추천
# -------------------------------
class ChatRecommendRequest(BaseModel):
    query: str

@app.post("/api/llm_recommend")
def llm_recommend(request: ChatRecommendRequest):
    result = llm_recommend_menu(request.query)
    return {"recommendation": result}

# -------------------------------
# 헬스 체크
# -------------------------------
@app.get("/")
def health_check():
    return {"message": "오늘의 먹방은 API 정상 동작 중"}
