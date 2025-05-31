import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from urllib.parse import quote
from webdriver_manager.chrome import ChromeDriverManager

# ✅ 브라우저 옵션 설정
options = webdriver.ChromeOptions()
options.add_argument("window-size=1280x1000")
options.add_argument("user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) "
                     "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1")

# ✅ WebDriver 설정
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ✅ CSV 파일 경로 설정
input_csv_path = 'C:/TodayMenu/청파동_키워드맛집.csv'
output_csv_path = 'C:/TodayMenu/청파동_키워드맛집_with_place_id.csv'

# ✅ CSV 파일 읽기
df = pd.read_csv(input_csv_path)

# ✅ place_id를 저장할 리스트
place_ids = []

# ✅ 각 가게 이름에 대해 place_id 추출
for index, row in df.iterrows():
    store_name = row['name']
    print(f"🔍 '{store_name}' 검색 중...")

    try:
        # 검색어 URL 인코딩
        encoded_name = quote(store_name)
        search_url = f'https://m.map.naver.com/search2/search.naver?query={encoded_name}'

        # 네이버 지도 모바일 검색 결과 페이지 접속
        driver.get(search_url)
        time.sleep(3)

        # 검색 결과에서 첫 번째 항목의 링크 추출
        first_result = driver.find_element(By.CSS_SELECTOR, 'ul.lst_site > li > a')
        href = first_result.get_attribute('href')

        # href에서 place_id 추출
        if 'place/' in href:
            place_id = href.split('place/')[1].split('?')[0]
            print(f"✅ '{store_name}'의 place_id: {place_id}")
            place_ids.append(place_id)
        else:
            print(f"❌ '{store_name}'의 place_id를 추출할 수 없습니다.")
            place_ids.append(None)

    except Exception as e:
        print(f"❌ '{store_name}'의 place_id 추출 실패: {e}")
        place_ids.append(None)
        continue

# ✅ place_id를 데이터프레임에 추가
df['place_id'] = place_ids

# ✅ 결과를 새로운 CSV 파일로 저장
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"\n🎉 place_id가 추가된 CSV 파일이 저장되었습니다: {output_csv_path}")

# ✅ 브라우저 종료
driver.quit()
