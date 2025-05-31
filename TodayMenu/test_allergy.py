import pandas as pd

# 데이터 불러오기
csv_path = "C:/TodayMenu/청파동_menu_nutrient.csv"
menu_df = pd.read_csv(csv_path)

# 알러지 및 건강 조건 정의
allergy_filter = "달걀"   # 달걀 알러지
sodium_limit = 800       # 고혈압 → 나트륨 800 이하

# 조건에 맞는 메뉴만 남기기
filtered_df = menu_df[
    (~menu_df["allergy"].str.contains(allergy_filter, na=False)) &
    (menu_df["sodium"] < sodium_limit)
]

# 결과 출력
print("추천 가능한 메뉴:")
print(filtered_df[["name", "sodium", "allergy"]])
