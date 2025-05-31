from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import SessionLocal, User, Menu, Restaurant, Review
from pydantic import BaseModel

router = APIRouter(prefix="/review", tags=["review"])

class ReviewRequest(BaseModel):
    username: str
    restaurant_id: int
    menu_id: int
    rating: int
    tags: list[str]
    comment: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ 가중치 계산 함수
def calculate_feedback_weight(rating: int, tags: list[str]) -> int:
    score = 0

    # 평점 기반 가중치
    if rating >= 4:
        score += 2
    elif rating <= 2:
        score -= 2

    # 태그 기반 가중치
    for tag in tags:
        if tag in ["좋아요", "가성비가 좋아요", "빨리 나와요", "접근성이 좋아요"]:
            score += 2
        elif tag == "별로예요":
            score -= 3

    return score

@router.post("")
def create_review(review: ReviewRequest, db: Session = Depends(get_db)):
    print("✅ 전달받은 username:", review.username)
    print("📥 받은 리뷰 데이터:", review.dict())

    user = db.query(User).filter(User.username == review.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    restaurant = db.query(Restaurant).filter(Restaurant.id == review.restaurant_id).first()
    if not restaurant:
        raise HTTPException(status_code=404, detail="Restaurant not found")

    menu = db.query(Menu).filter(Menu.id == review.menu_id).first()
    if not menu:
        raise HTTPException(status_code=404, detail="Menu not found")

    existing = db.query(Review).filter_by(
        user_id=user.id,
        restaurant_id=review.restaurant_id,
        menu_id=review.menu_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="이미 리뷰를 작성했습니다")

    new_review = Review(
        user_id=user.id,
        restaurant_id=review.restaurant_id,
        menu_id=review.menu_id,
        rating=review.rating,
        tags=",".join(review.tags),
        comment=review.comment
    )
    db.add(new_review)

    # ✅ 리뷰 기반 가중치 계산 및 Menu에 반영
    weight = calculate_feedback_weight(review.rating, review.tags)
    if menu.feedback_score is None:
        menu.feedback_score = 0
    menu.feedback_score += weight

    db.commit()
    db.refresh(new_review)

    return {"message": "리뷰 저장 완료!"}

@router.get("/check")
def check_review(username: str, restaurant_id: int, menu_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"exists": False}

    review = db.query(Review).filter_by(
        user_id=user.id,
        restaurant_id=restaurant_id,
        menu_id=menu_id
    ).first()

    if review:
        return {
            "exists": True,
            "review": {
                "rating": review.rating,
                "tags": review.tags.split(",") if review.tags else [],
                "comment": review.comment
            }
        }
    return {"exists": False}
