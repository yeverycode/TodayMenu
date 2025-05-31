import pandas as pd

# 파일 경로 설정
original_path = r"C:\TodayMenu\backend\data\final_menu_data.csv"  # 기존 데이터
new_data_path = r"C:\TodayMenu\final_menu_data_with_tags_fuzzy_reordered.csv"  # 새로 추가할 데이터
output_path = r"C:\TodayMenu\final_combined.csv"  # 병합된 결과 저장

# 기존 데이터와 새 데이터 불러오기
df_old = pd.read_csv(original_path)
df_new = pd.read_csv(new_data_path)

# 현재 최대 menu_id와 restaurant_id 찾기
max_menu_id = df_old["menu_id"].max()
max_restaurant_id = df_old["restaurant_id"].max()

# 기존 식당 키(이름 + 주소)로 중복 여부 확인
restaurant_key_old = df_old[["place_name", "address", "restaurant_id"]].drop_duplicates()
restaurant_key_new = df_new[["place_name", "address"]].drop_duplicates()

# 새로운 식당에 restaurant_id 할당
new_restaurant_ids = {}
for _, row in restaurant_key_new.iterrows():
    key = (row["place_name"], row["address"])
    match = restaurant_key_old[
        (restaurant_key_old["place_name"] == key[0]) &
        (restaurant_key_old["address"] == key[1])
    ]
    if not match.empty:
        new_restaurant_ids[key] = match.iloc[0]["restaurant_id"]
    else:
        max_restaurant_id += 1
        new_restaurant_ids[key] = max_restaurant_id

# menu_id 부여
df_new = df_new.copy()
df_new["menu_id"] = range(max_menu_id + 1, max_menu_id + 1 + len(df_new))

# restaurant_id 할당
df_new["restaurant_id"] = df_new.apply(
    lambda row: new_restaurant_ids[(row["place_name"], row["address"])],
    axis=1
)

# 기존 데이터와 새 데이터 병합
final_df = pd.concat([df_old, df_new], ignore_index=True)

# 결과 저장
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
