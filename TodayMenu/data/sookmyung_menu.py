import requests
import csv
import json
import time

# 카카오 API 설정
API_KEY = "815a330dcfb69987a6c219836b68598c"
headers = {"Authorization": f"KakaoAK {API_KEY}"}

# 동별 중심 좌표 설정
areas = {
    "갈월동": (126.9722,37.54550),
    "청파동": (126.9668, 37.5450),
    "효창동": (126.9800, 37.5635),
    "남영동": (126.9737, 37.5427)
}

#  공통 키워드
queries = ["술집", "호프", "식당", "맛집", "카페", "주점", "포차", "양식", "한식", "중식", "일식", "퓨전", "와인바", "펍", "맥주"]

# 키워드 검색 함수
def search_keyword_places(query, x, y, radius=3000, page=1):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    params = {
        "query": query,
        "x": x,
        "y": y,
        "radius": radius,
        "page": page,
        "size": 15
    }
    res = requests.get(url, headers=headers, params=params)
    print(f"[DEBUG] {query} - Page {page} - status: {res.status_code}")
    try:
        return res.json()
    except Exception as e:
        print("[ERROR] JSON 디코딩 실패:", e)
        return {}

# 크롤링 함수 (동 필터링 포함)
def crawl_keyword_places(queries, area_name, x, y, radius=3000, max_count=50):
    collected = []
    seen_ids = set()
    
    for query in queries:
        for page in range(1, 100):
            result = search_keyword_places(query, x, y, radius, page)
            documents = result.get("documents", [])
            print(f"[{area_name}] {query} - Page {page} - 수신된 가게 수: {len(documents)}")
            if not documents:
                break
            for place in documents:
                pid = place.get("id")
                address = place.get("address_name", "")

                # 주소 필터링 (동 이름 포함 여부)
                if area_name.replace("동", "") not in address:
                    continue

                if pid not in seen_ids:
                    seen_ids.add(pid)
                    collected.append(place)

                    # 50개 채우면 종료
                    if len(collected) >= max_count:
                        return collected
            time.sleep(0.3)
    
    return collected

# CSV 저장
def save_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["name", "address", "phone", "category", "lat", "lng", "id", "url"])
        for place in data:
            writer.writerow([
                place.get('place_name', ''),
                place.get('address_name', ''),
                place.get('phone', ''),
                place.get('category_name', ''),
                place.get('y', ''),
                place.get('x', ''),
                place.get('id', ''),
                place.get('place_url', '')
            ])

# JSON 저장
def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 실행
if __name__ == "__main__":
    for area_name, (x, y) in areas.items():
        print(f"\n🔍 [{area_name}] 식당 및 술집 수집 시작...")
        places = crawl_keyword_places(queries, area_name, x, y, radius=3000, max_count=50)
        print(f"[{area_name}] 수집 완료: {len(places)}개")

        save_to_csv(places, f"{area_name}_final식당_50개.csv")
        save_to_json(places, f"{area_name}_final식당_50개.json")
        print(f"[{area_name}] 저장 완료 (CSV & JSON)")
