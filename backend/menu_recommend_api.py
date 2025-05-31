from fastapi import APIRouter, Depends
from pydantic import BaseModel
import pandas as pd
import random
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import SessionLocal
from models import RecommendationHistory

router = APIRouter()

# CSV 전처리
menu_df = pd.read_csv("./data/final_menu_data.csv")
menu_df["menu_price"] = menu_df["menu_price"].astype(int)
menu_df["region"] = menu_df["region"].str.strip()
menu_df["allergy"] = menu_df["allergy"].apply(lambda x: [] if pd.isna(x) else [a.strip() for a in str(x).split(",")])
menu_df["ingredient"] = menu_df["ingredient"].replace("없음", "").fillna("")
menu_df["top_tags"] = menu_df["top_tags"].fillna("")
menu_df["disease"] = menu_df["disease"].apply(eval)

if "menu_id" not in menu_df.columns or "restaurant_id" not in menu_df.columns:
    raise ValueError("menu_id 또는 restaurant_id 컬럼이 없습니다.")


# 위험 재료 매핑
DISEASE_DANGER_FOODS = {
    "당뇨": [
        "설탕", "시럽", "케이크", "케익", "디저트", "와플", "라떼", "빙수", "단호박죽", "초코", "빵", "젤리", "쿠키",
        "꿀", "달고나", "연유", "버터", "아이스크림", "도넛", "마카롱", "초콜릿바", "잼", "카라멜", "푸딩",
        "팬케이크", "떡", "군고구마", "감자", "옥수수", "밀크티", "베이글", "라면", "우동", "쌀국수", "피자",
        "요거트(가당)", "에너지바", "시리얼", "과일주스", "말린과일", "무화과", "바나나"
    ],
    "고혈압": [
        "짠", "소금", "라면", "찌개", "간장", "국물", "짬뽕", "된장", "김치", "불고기", "짜장",
        "제육", "곰탕", "육개장", "감자탕", "순대국", "돼지국밥", "해장국", "삼겹살", "닭갈비", "마라탕",
        "간장계란밥", "쌈장", "어묵탕", "우동", "짬짜면", "비빔면", "소세지볶음", "햄", "베이컨"
    ],
    "저혈압": [
        "카페인", "커피", "아메리카노", "에스프레소", "콜드브루", "카푸치노", "더치커피", "커피우유",
        "라떼", "녹차", "말차", "콜라", "펩시", "사이다", "스프라이트", "에너지드링크", "몬스터", "바나나", 
        "감자전", "토마토", "수박", "죽", "수박 주스", "떡볶이", "국수", "샐러드", "오이냉국", "수박화채", "포케", "케이크", "도넛"
    ],
    "신장질환": [
        "국물", "젓갈", "김치", "명란", "어묵", "햄", "소세지", "쏘세지", "고기", "돼지고기", "소고기", "닭고기", "불고기",
        "절임", "찌개", "탕", "꼬치", "불고기", "제육", "소불백", "돼지불백", "닭", "돈까스", "스테이크", 
        "닭도리탕", "순대", "갈비", "양념갈비", "삼겹살", "족발", "보쌈", "차슈", "햄버거", "로스트", "토시살", 
        "껍데기", "꼬들살", "소", "돼지", "계육", "닭갈비", "닭볶음탕", "닭강정", "양념치킨", "후라이드", 
        "곱창", "대창", "막창", "갈매기살", "항정살", "반계탕", "계", "치킨", "소갈비탕", "갈비탕", "곰탕", 
        "설렁탕", "육개장", "순대국", "소머리국밥", "돼지국밥", "꼬들", "목살", "함박", "불백", "삼계탕", 
        "백숙", "닭죽", "육회", "떡갈비", "육사시미", "간", "돼지껍데기", "소시지", "안심", "등심", 
        "채끝", "안창살", "차돌박이", "우설", "육포", "햄버거 패티", "소꼬리", "수육", "바베큐", 
        "그릴드 치킨", "BBQ", "염통", "차돌", "램", "윙", "히레", "깐풍기", "닭발", "마파두부", 
        "두부", "콩", "빵", "파스타", "스파게티", "면", "누들", "크로와상", "라면", "우동", 
        "만두", "피자", "떡국", "짜장", "짬뽕", "튀김", "꽃빵", "탕수육", "샌드위치", "떡볶이", 
        "로제", "수제비", "팬케이크", "러스크", "와플", "베이글", "부대찌개", "마라탕", "국수", 
        "짜장면", "쟁반짜장", "볶음면", "브리또", "타코야키", "비빔면", "칼국수", "잔치국수", 
        "콘크림파스타", "카르보나라", "알리오올리오", "찐빵", "깜파뉴", "까스", "호떡", "도넛", 
        "쿠키", "카스테라", "파이", "토스트", "프레첼", "식빵", "바게트", "꿔바로우", 
        "포카치아", "피타", "나쵸", "타코", "통밀", "밀가루", "핫도그", "가스", "동", 
        "떡", "다시마국수", "쫄면", "동치미국수", "온면", "초계국수", "돈가스", "버거", 
        "라멘", "짜빠구리", "새우장"
    ]
}
HUNGER_FOOD_CATEGORIES = {
    "적음": ["샐러드", "요거트", "버블티", "샌드위치", "김밥", "주먹밥", "떡볶이", "라면", "쿠키", "머핀", "과일컵", "밀크티", "크로와상", "베이글"],
    "많이": ["피자", "치킨", "덮밥", "찌개", "고기", "삼겹살", "갈비", "불고기", "국수", "비빔밥", "짜장면", "짬뽕", "탕수육", "부대찌개", "김치찌개", "된장찌개", "삼계탕", "돈까스", "스테이크", "햄버거"]
}

DRINK_PAIRINGS = {
    "소주": ["삼겹살", "족발", "찌개", "전", "보쌈", "홍어", "곱창", "막창", "육회", "김치찌개", "오징어볶음", "두부김치", "닭발", "조개탕", "회"],
    "맥주": ["치킨", "피자", "감자튀김", "소시지", "족발", "파전", "치즈스틱", "핫도그", "양념치킨", "오징어튀김", "매운탕", "짬뽕", "불닭볶음면", "비빔면"],
    "와인": ["치즈", "파스타", "스테이크", "샐러드", "리조또", "카나페", "훈제오리", "감바스", "바게트", "프로슈토", "연어", "초밥"],
    "막걸리": ["파전", "빈대떡", "김치전", "두부김치", "보쌈", "홍어", "생선구이", "부침개", "감자전", "동동주"],
    "하이볼": ["회", "돈까스", "야키토리", "규카츠", "감바스", "타코", "카레"]
}

# 비선호만 필터링
def filter_by_disliked_only(df, disliked):
    def contains_disliked(row):
        combined = f"{row['menu_name']} {row['ingredient']}".lower()
        return any(d.lower() in combined for d in disliked)
    return df[~df.apply(contains_disliked, axis=1)]

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class MenuRecommendInput(BaseModel):
    user_id: int
    region: str
    alone: str
    budget: str
    drink: str
    hunger: str
    allergies: list[str]
    diseases: list[str]
    liked_ingredients: list[str] = []
    disliked_ingredients: list[str] = []

def no_menu_response(msg):
    return {
        "menu_name": msg,
        "place_name": "-",
        "menu_price": "-",
        "distance": "-",
        "address": "-",
        "url": "-",
        "user_id": None,
        "menu_id": None,
        "restaurant_id": None,
        "reason": {}
    }

@router.post("/menu-recommend")
def recommend_menu(input_data: MenuRecommendInput, db: Session = Depends(get_db)):
    try:
        # 1. 예산 필터
        price_min, price_max = 0, 999999
        if "1만원 미만" in input_data.budget:
            price_max = 10000
        elif "1~2만원" in input_data.budget:
            price_min, price_max = 10000, 20000
        elif "2~3만원" in input_data.budget:
            price_min, price_max = 20000, 30000
        elif "3~4만원" in input_data.budget:
            price_min, price_max = 30000, 40000
        elif "4만원 이상" in input_data.budget:
            price_min = 40000

        filtered = menu_df[
            (menu_df["region"] == input_data.region) &
            (menu_df["menu_price"] >= price_min) &
            (menu_df["menu_price"] <= price_max)
        ]
        if filtered.empty:
            return no_menu_response("예산에 맞는 메뉴가 없습니다")

        # 2. 알러지 필터
        filtered = filtered[filtered["allergy"].apply(lambda x: not any(a in x for a in input_data.allergies))]
        if filtered.empty:
            return no_menu_response("알러지를 고려했을 때 추천할 수 없습니다")

        # 3. 지병 필터
        def is_safe(row):
            combined = f"{row['menu_name']} {row['ingredient']} {row['top_tags']}".lower()
            for disease in input_data.diseases:
                for danger in DISEASE_DANGER_FOODS.get(disease, []):
                    if danger.lower() in combined:
                        return False
            return True
        filtered = filtered[filtered.apply(is_safe, axis=1)]
        if filtered.empty:
            return no_menu_response("지병을 고려했을 때 추천할 수 없습니다")

        # 4. DB에서 선호/비선호 메뉴 불러오기
        user_prefs = db.execute(
            text("SELECT preference_type, menu_name FROM user_preference WHERE user_id = :uid"),
            {"uid": input_data.user_id}
        ).fetchall()
        liked = list(set(input_data.liked_ingredients + [r[1] for r in user_prefs if r[0] == "선호"]))
        disliked = list(set(input_data.disliked_ingredients + [r[1] for r in user_prefs if r[0] == "비선호"]))

        # 5. 싫어요/낮은 점수 제외
        bads = db.query(RecommendationHistory).filter(
            RecommendationHistory.user_id == input_data.user_id,
            (RecommendationHistory.feedback == "bad") |
            ((RecommendationHistory.review_score != None) & (RecommendationHistory.review_score <= 2))
        ).all()
        bad_menu_names = [b.menu_name for b in bads]
        filtered = filtered[~filtered["menu_name"].isin(bad_menu_names)]

        # 6. 비선호 필터링
        filtered = filter_by_disliked_only(filtered, disliked)
        if filtered.empty:
            return no_menu_response("비선호 재료를 고려했을 때 추천할 수 없습니다")

        # 7. 피드백/리뷰 기반 가중치
        good_feedbacks = db.query(RecommendationHistory).filter(
            RecommendationHistory.user_id == input_data.user_id,
            RecommendationHistory.feedback == "good"
        ).all()
        positive_names = [g.menu_name for g in good_feedbacks]

        high_reviews = db.query(RecommendationHistory).filter(
            RecommendationHistory.user_id == input_data.user_id,
            RecommendationHistory.review_score >= 4
        ).all()
        high_review_names = [h.menu_name for h in high_reviews]

        candidates = []
        for _, row in filtered.iterrows():
            weight = 1
            combined = f"{row['menu_name']} {row['ingredient']}".lower()
            if any(l.lower() in combined for l in liked):
                weight += 2
            if row["menu_name"] in positive_names:
                weight += 3
            if row["menu_name"] in high_review_names:
                weight += 2
            candidates += [row] * weight

        if not candidates:
            return no_menu_response("조건에 맞는 후보가 없습니다")

        selected = random.choice(candidates)
        selected_ingredients = [i.strip().lower() for i in str(selected["ingredient"]).split(",")]

        reason = {
            "liked": [kw for kw in liked if any(kw.lower() in ing for ing in selected_ingredients)],
            "disliked": [kw for kw in disliked if any(kw.lower() in ing for ing in selected_ingredients)],
            "allergic": [kw for kw in input_data.allergies if any(kw.lower() in ing for ing in selected_ingredients)],
            "feedback_match": selected["menu_name"] in positive_names,
            "disease_safe": True
        }

        db.add(RecommendationHistory(
            user_id=input_data.user_id,
            place_name=selected["place_name"],
            menu_name=selected["menu_name"],
            menu_id=int(selected["menu_id"]),
            restaurant_id=int(selected["restaurant_id"])
        ))
        db.commit()

        return {
            "menu_name": selected["menu_name"],
            "place_name": selected["place_name"],
            "menu_price": selected["menu_price"],
            "distance": "도보 10분 이내",
            "address": selected["address"],
            "url": selected["url"],
            "user_id": input_data.user_id,
            "menu_id": selected["menu_id"],
            "restaurant_id": selected["restaurant_id"],
            "reason": reason
        }

    except Exception as e:
        print("추천 오류:", e)
        return no_menu_response("추천 중 오류 발생")
