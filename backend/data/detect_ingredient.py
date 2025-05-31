import pandas as pd
import csv

# 파일 경로
input_path = "C:/TodayMenu/gal_menu_data.csv"
output_path = "C:/TodayMenu/gal_menu_data.csv"

# 식재료 키워드 정의
ingredient_keywords = {
    "고기": [ "불고기", "제육", "소불백", "돼지불백", "닭", "돈까스", "스테이크", "햄", "베이컨", "닭도리탕", 
        "순대", "갈비", "양념갈비", "삼겹살", "족발", "보쌈", "차슈", "햄버거", "로스트", "토시살", "삼겹", "오겹", "껍데기", "꼬들살",
        "소", "돼지", "계육", "닭갈비", "닭볶음탕", "닭강정", "양념치킨", "후라이드", "곱창", "대창", "막창", "갈매기살", "항정살",
        "반계탕", "계", "치킨", "소갈비탕", "갈비탕", "곰탕", "설렁탕", "육개장", "순대국", "소머리국밥", "돼지국밥",
        "꼬들", "목살", "함박", "불백", "삼계탕", "백숙", "닭죽", "육회", "떡갈비", "육사시미", "간", "돼지껍데기",
        "소시지", "안심", "등심", "채끝", "안창살", "차돌박이", "우설", "육포", "햄버거 패티", "소꼬리", "수육",
        "바베큐", "그릴드 치킨", "BBQ", "염통", "차돌", "램", "윙", "히레", "깐풍기"
    ],
    "버섯": [
        "버섯", "표고버섯", "새송이버섯", "양송이버섯", "팽이버섯", "느타리버섯", "백만송이버섯", "송이버섯", "표고", "새송이", "양송이", "팽이", "느타리","머쉬룸"
    ],
    "고수": [
        "고수", "cilantro", "샹차이", "향채", "코리앤더", "고수잎", "고수나물", "고수소스", "쌀국수"
    ],
    "내장": [
        "내장", "곱창", "대창", "막창", "염통", "간", "천엽", "양깃머리", "선지", "염통꼬치", "소내장", "돼지내장", "염통구이", "내장탕", "곱창전골", "내장볶음"
    ],
    "닭발": [
        "닭발", "무뼈닭발", "매운닭발", "석쇠닭발", "닭발볶음", "닭발구이", "직화닭발", "불닭발"
    ],
    "해산물": [
        "해산물", "해물", "새우", "오징어", "문어", "게", "꽃게", "대게", "킹크랩", "랍스터", "전복", "조개", "가리비",
        "홍합", "멍게", "낙지", "해물탕", "해물파전", "오징어볶음", "해물볶음밥", "해물라면", "해물덮밥", "해물우동",
        "칠리새우", "버터새우", "감바스", "쉬림프", "크랩", "간장게장", "양념게장", "바지락", "모듬해물", "스파이시해물", "해물찜"
    ]
}

# 포함된 재료 감지 함수
def detect_ingredients(menu_name):
    detected = []
    for ingredient, keywords in ingredient_keywords.items():
        if any(keyword in str(menu_name) for keyword in keywords):
            detected.append(ingredient)
    return ', '.join(detected) if detected else "없음"

# CSV 불러오기
df = pd.read_csv(input_path)

# 불필요한 컬럼 삭제 (존재할 경우만)
columns_to_drop = ["ingredients", "all_sensitive_ingredients"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# ingredient 컬럼 생성
df["ingredient"] = df["menu_name"].apply(detect_ingredients)

# 저장
df.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
print(f"✅ 저장 완료: {output_path}")
