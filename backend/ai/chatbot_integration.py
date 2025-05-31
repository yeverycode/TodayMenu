from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from models import SessionLocal, User, Feedback
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import pandas as pd
import traceback
from collections import Counter
import os
import time

# -------------------- Load Environment Variables --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")

# -------------------- Create Router --------------------
router = APIRouter(prefix="/api")

# -------------------- Database Session Dependency --------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------- Load CSV Menu Data --------------------
menu_df = pd.read_csv("./data/final_menu_data.csv")
menu_df["disease"] = menu_df["disease"].apply(eval)

# -------------------- Map Situation Keywords to Emotional Tags --------------------
def extract_situation_tags(situation: str) -> list[str]:
    mapping = {
        "매콤": ["매움"],
        "해장": ["속풀이"],
        "꿀꿀": ["단맛", "중독성"],
        "가볍": ["가성비", "양많음"],
        "친구": ["분위기좋음", "특별함"],
        "추워": ["속풀이"],
        "덥": ["가성비"],
        "달달": ["단맛"],
        "배고": ["양많음"],
        "야식": ["야식"]
    }
    tags = set()
    for keyword, tag_list in mapping.items():
        if keyword in situation:
            tags.update(tag_list)
    return list(tags)

# -------------------- Filter Out Irrelevant Input --------------------
def is_irrelevant_input(situation: str) -> bool:
    irrelevant_keywords = ["심심", "ㅋㅋ", "ㅎㅎ", "하하", "재미", "뭐하지", "놀자"]
    return any(keyword in situation.lower() for keyword in irrelevant_keywords)

# -------------------- Filter Menus by Disease --------------------
def filter_menu_by_disease(df: pd.DataFrame, diseases: list[str]) -> pd.DataFrame:
    return df[~df['disease'].apply(lambda risks: any(d in risks for d in diseases))]

# -------------------- Apply Feedback Weights --------------------
def apply_feedback_weights(df: pd.DataFrame, db: Session) -> pd.DataFrame:
    feedbacks = db.query(Feedback).all()
    good_counts = Counter((f.place_name, f.menu_name) for f in feedbacks if f.feedback == "good")
    bad_counts = Counter((f.place_name, f.menu_name) for f in feedbacks if f.feedback == "bad")

    def compute_score(row):
        key = (row["place_name"], row["menu_name"])
        return good_counts[key] - bad_counts[key]

    df = df.copy()
    df["feedback_score"] = df.apply(compute_score, axis=1)
    return df.sort_values(by="feedback_score", ascending=False)

# -------------------- Get Relevant Menus (토큰 절약용) --------------------
def get_relevant_menus(df: pd.DataFrame, situation_tags: list[str], max_count: int = 15) -> pd.DataFrame:
    """상황에 맞는 메뉴만 선별하여 토큰 절약"""
    if situation_tags:
        relevant = df[df["top_tags"].apply(
            lambda x: any(tag in str(x) for tag in situation_tags)
        )].head(max_count)
        if not relevant.empty:
            return relevant
    return df.head(max_count)

# -------------------- GPT-based Recommendation System --------------------
class MenuRecommendationSystem:
    def __init__(self, api_key, menu_list):
        self.menu_list = menu_list
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            api_key=api_key,
            streaming=True,
            max_tokens=200   # 토큰 사용량 줄임
        )
        self.conversation_stores = {}

        # 프롬프트 간소화
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""
오늘의 메뉴 추천 AI입니다.

규칙:
- 한국어로만 답변
- 아래 메뉴에서만 추천
- 짧고 감성적인 이유 포함
- 이전 대화 무시, 현재 입력만 사용

추천 메뉴:
{self.menu_list}

사용자 정보: {{user_profile}}
날씨: {{weather}}
"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

    def get_session_history(self, session_id: str):
        if session_id not in self.conversation_stores:
            self.conversation_stores[session_id] = InMemoryChatMessageHistory()
        return self.conversation_stores[session_id]

    def create_conversation_chain(self):
        chain = self.prompt_template | self.llm
        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

# -------------------- Streaming Recommendation API --------------------
@router.get("/llm-recommend-stream")
async def llm_recommend_stream(
    user_id: int,
    weather: str,
    situation: str,
    db: Session = Depends(get_db)
):
    try:
        if is_irrelevant_input(situation):
            async def irrelevant_response():
                yield f"data: 저는 메뉴 추천을 해주는 챗봇이에요. 메뉴 관련 질문을 해주세요.\n\n"
                yield f"data: [END]\n\n"

            return StreamingResponse(irrelevant_response(), media_type="text/event-stream")

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("존재하지 않는 사용자입니다.")

        allergies = [a.allergy for a in user.allergies]
        likes = [p.menu_name for p in user.preferences if p.preference_type == "선호"]
        dislikes = [p.menu_name for p in user.preferences if p.preference_type == "비선호"]
        user_diseases = [d.disease for d in user.diseases]

        # 사용자 프로필 간소화
        user_profile = f"알레르기: {', '.join(allergies[:3]) if allergies else '없음'} / 선호: {', '.join(likes[:3]) if likes else '없음'} / 비선호: {', '.join(dislikes[:3]) if dislikes else '없음'}"

        safe_menu_df = filter_menu_by_disease(menu_df, user_diseases)
        scored_menu_df = apply_feedback_weights(safe_menu_df, db)

        situation_tags = extract_situation_tags(situation)
        
        # 관련 메뉴만 선별 (토큰 절약)
        relevant_menus = get_relevant_menus(scored_menu_df, situation_tags, max_count=15)
        
        # 메뉴 리스트 간소화 (URL 제거, 감성 태그 간략화)
        menu_list_str = "\n".join(
            f"- {row['menu_name']} ({row['place_name']})"
            for _, row in relevant_menus[['place_name', 'menu_name']]
            .drop_duplicates()
            .iterrows()
        )

        recommendation_system = MenuRecommendationSystem(OPENAI_API_KEY, menu_list_str)
        conversation = recommendation_system.create_conversation_chain()

        async def generate():
            try:
                if user_diseases:
                    yield f"data: {', '.join(user_diseases)}에 따라 안전한 메뉴로 추천해드릴게요.\n\n"
                else:
                    yield f"data: 상황에 맞는 메뉴를 추천해드릴게요.\n\n"

                session_id = f"streaming_session_{user_id}_{int(time.time())}"

                # 입력 데이터 간소화
                input_data = {
                    "input": situation,
                    "user_profile": user_profile,
                    "weather": weather[:20] if weather else "정보없음"  # 날씨 정보도 간략화
                }

                response_generator = conversation.astream(
                    input_data,
                    config={"configurable": {"session_id": session_id}}
                )

                async for chunk in response_generator:
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if content:  # 빈 내용 필터링
                        yield f"data: {content}\n\n"

                yield f"data: [END]\n\n"
                
            except Exception as e:
                print(f"Generation error: {str(e)}")
                traceback.print_exc()
                yield f"data: 추천 중 오류가 발생했어요. 다시 시도해 주세요.\n\n"
                yield f"data: [END]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        print(f"Main error: {str(e)}")
        traceback.print_exc()

        async def error_response():
            yield f"data: 서버 오류가 발생했습니다.\n\n"
            yield f"data: [END]\n\n"

        return StreamingResponse(error_response(), media_type="text/event-stream")