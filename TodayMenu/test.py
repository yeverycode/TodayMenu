import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors

# 데이터 불러오기
csv_path = "C:\TodayMenu\청파동_menu_price.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

df['menu_price'] = pd.to_numeric(df['menu_price'], errors='coerce')
df['menu_price'] = df['menu_price'].fillna(0)

features = df[['category', 'weather', 'menu_price']].copy()

encoder = OneHotEncoder()
encoded = encoder.fit_transform(features[['category', 'weather']]).toarray()
menu_price = features['menu_price'].values.reshape(-1, 1)
X = np.hstack([encoded, menu_price])

model = NearestNeighbors(n_neighbors=3, metric='euclidean')
model.fit(X)

def recommend(menu_idx):
    distances, indices = model.kneighbors([X[menu_idx]])
    print("\n[추천 결과]")
    for idx, distance in zip(indices[0], distances[0]):
        place = df.iloc[idx]['place_name']
        menu = df.iloc[idx]['menu_name']
        price = df.iloc[idx]['menu_price']
        weather = df.iloc[idx]['weather']
        print(f"📌 {place} - {menu} ({price}원), 날씨: {weather}, 거리: {distance:.2f}")

# ---------------------------
# 자연어 -> 조건 매핑
weather_map = {
    "추운": "추움",
    "따뜻한": "따뜻함",
    "더운": "더움",
    "쌀쌀한": "쌀쌀",
    "비오는": "비",
    "흐린": "흐림"
}

def get_weather_condition(user_input):
    for key in weather_map:
        if key in user_input:
            return weather_map[key]
    return None

# ---------------------------
# 사용자 입력
user_input = input("메뉴를 추천받고 싶으면 말씀해주세요 (예: 추운 날씨에 먹을 음식): ")

condition = get_weather_condition(user_input)

if condition:
    filtered_df = df[df['weather'].str.contains(condition)]
    
    if not filtered_df.empty:
        random_idx = filtered_df.sample(1).index[0]
        recommend(random_idx)
    else:
        print("추천할 메뉴가 없어요.")
else:
    print("조건에 맞는 날씨가 없어요.")
