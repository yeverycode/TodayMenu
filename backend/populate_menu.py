# backend/populate_menu.py

import pandas as pd
from database import SessionLocal
from models import Menu

# CSV 로드
df = pd.read_csv("./data/final_menu_data.csv")
db = SessionLocal()

for _, row in df.iterrows():
    menu = Menu(
        id=int(row['menu_id']),
        place_name=row['place_name'],
        region=row['region'],
        menu_name=row['menu_name'],
        price=int(row['menu_price']),
        address=row['address'],
        url=row['url'],
        allergy=str(row['allergy']) if pd.notna(row['allergy']) else "",
        disease=str(row['disease']) if pd.notna(row['disease']) else "",
        emotion_summary=str(row['emotion_summary']) if pd.notna(row['emotion_summary']) else "",
        top_tags=str(row['top_tags']) if pd.notna(row['top_tags']) else "",
        ingredient=str(row['ingredient']) if pd.notna(row['ingredient']) else "",
        category="",   # 필요 시 추가
        weather="",    # 필요 시 추가
        restaurant_id=int(row['restaurant_id'])
    )
    db.add(menu)


db.commit()
db.close()
print("✅ 메뉴 데이터 삽입 완료")
