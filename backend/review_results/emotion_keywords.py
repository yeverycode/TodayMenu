import os
import pandas as pd

# 경로 설정
menu_csv_path = r"C:\TodayMenu\갈월동_메뉴_통합.csv"
review_folder = r"C:\TodayMenu\review_results"
output_path = r"C:\TodayMenu\갈월동_메뉴_통합_결과.csv"

# 감성 키워드 사전
emotion_keywords = {
    "매움": ["맵", "매콤", "얼얼", "불맛"],
    "단맛": ["달아", "달콤", "단맛"],
    "짠맛": ["짭", "짠맛"],
    "양많음": ["푸짐", "양이 많", "듬뿍"],
    "가성비": ["가성비", "가격대비", "혜자", "저렴"],
    "재방문": ["또 가", "다시 가", "재방문", "생각나", "단골"],
    "친절함": ["친절", "서비스 좋", "응대"],
    "분위기좋음": ["분위기", "깔끔", "조용", "청결", "인테리어", "운치"],
    "속풀이": ["해장", "속이 풀"],
    "야식": ["야식", "늦게", "밤에"],
    "중독성": ["중독", "계속", "자꾸", "생각나"],
    "특별함": ["특이", "독특", "색다른", "별미", "섹시푸드"],
    "냄새없음": ["잡내", "냄새 안"],
    "포장추천": ["포장", "배달"]
}

# 감성 태그 추출 함수
def extract_emotion_tags(text):
    tags = set()
    for tag, keywords in emotion_keywords.items():
        for kw in keywords:
            if kw in text:
                tags.add(tag)
                break
    return list(tags)

# 리뷰 파일 목록 불러오기
all_review_files = os.listdir(review_folder)

# 메뉴 통합 CSV 불러오기
df = pd.read_csv(menu_csv_path)
df["emotion_summary"] = ""
df["top_tags"] = [[] for _ in range(len(df))]

# 각 식당에 대해 리뷰 파일 매칭
for idx, row in df.iterrows():
    restaurant_id = str(row["restaurant_id"])
    place_name = str(row["place_name"]).replace(" ", "")

    # 1. restaurant_id 기반 검색
    matched_file = next(
        (f for f in all_review_files if f.endswith(f"_{restaurant_id}.csv")),
        None
    )

    # 2. 없으면 place_name 기반 검색
    if not matched_file:
        matched_file = next(
            (f for f in all_review_files if f.startswith(f"reviews_{place_name}")),
            None
        )

    if matched_file:
        try:
            review_path = os.path.join(review_folder, matched_file)
            review_df = pd.read_csv(review_path)

            if "review" not in review_df.columns:
                print(f"❌ 'review' 컬럼 없음: {matched_file}")
                df.at[idx, "top_tags"] = ["리뷰없음"]
                df.at[idx, "emotion_summary"] = "리뷰없음"
                continue

            review_df = review_df.dropna(subset=["review"])
            review_df = review_df[~review_df["review"].isin(["더보기", "내꺼"])]

            combined_reviews = " ".join(str(r) for r in review_df["review"].tolist())
            tags = extract_emotion_tags(combined_reviews)

            df.at[idx, "top_tags"] = tags if tags else ["리뷰있음_태그없음"]
            df.at[idx, "emotion_summary"] = ", ".join(tags) if tags else "리뷰있음_태그없음"

        except Exception as e:
            print(f"❌ 오류 - {matched_file}: {e}")
            df.at[idx, "top_tags"] = ["리뷰오류"]
            df.at[idx, "emotion_summary"] = "리뷰오류"
    else:
        df.at[idx, "top_tags"] = ["리뷰없음"]
        df.at[idx, "emotion_summary"] = "리뷰없음"
        print(f"⚠️ 리뷰 파일 없음 (restaurant_id: {restaurant_id}, place_name: {place_name})")

# 결과 저장
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 리뷰 감성 태그 반영 완료 → {output_path}")
