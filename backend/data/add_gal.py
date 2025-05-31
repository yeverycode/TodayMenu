import pandas as pd

# ✅ 파일 경로
menu_path = "C:/TodayMenu/gal_menu_data.csv"
info_path = "C:/TodayMenu/갈월동식당.csv"

# ✅ CSV 파일 불러오기
menu_df = pd.read_csv(menu_path)
info_df = pd.read_csv(info_path)

# ✅ 컬럼명 맞추기: name → place_name
info_df = info_df.rename(columns={"name": "place_name"})

# ✅ region 추가
menu_df["region"] = "갈월동"

# ✅ 주소 및 URL 병합
merged_df = pd.merge(menu_df, info_df[["place_name", "address", "url"]], on="place_name", how="left")

# ✅ 감정 요약 및 태그 기본값
merged_df["emotion_summary"] = ""
merged_df["top_tags"] = ""

# ✅ menu_id와 restaurant_id 생성
merged_df["menu_id"] = range(len(merged_df))
restaurant_id_map = {name: idx for idx, name in enumerate(merged_df["place_name"].unique())}
merged_df["restaurant_id"] = merged_df["place_name"].map(restaurant_id_map)

# ✅ 최종 열 순서 정리
final_df = merged_df[
    [
        "place_name", "region", "menu_name", "menu_price", "address", "url",
        "allergy", "disease", "emotion_summary", "top_tags",
        "menu_id", "restaurant_id", "ingredient"
    ]
]

# ✅ CSV로 저장
final_df.to_csv("C:/TodayMenu/갈월동_메뉴_통합.csv", index=False, encoding="utf-8-sig")
print("✅ 갈월동_메뉴_통합.csv 저장 완료!")
