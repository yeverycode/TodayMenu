# 🍽️ 오늘의 먹방

> **AI 기반 개인 맞춤형 메뉴 추천 시스템**  
> 숙명여자대학교 IPS 대회 출품작 (2025.04.01 ~ 2025.05.30)

사용자의 건강 정보와 취향을 고려하여 최적의 메뉴를 추천하는 AI 기반 앱입니다.

---

## 📖 목차

- [팀 소개](#-팀-소개)
- [프로젝트 개요](#-프로젝트-개요)
- [주요 기능](#-주요-기능)
- [추천 알고리즘](#-추천-알고리즘)
- [기술 스택](#️-기술-스택)
- [시스템 아키텍처](#️-시스템-아키텍처)
- [주요 화면](#-주요-화면)

---

## 👥 팀 소개

| 이름 | 역할 |
|------|------|
| **조예인** (팀장) | 추천 시스템 알고리즘 개발, 청파동 데이터 수집 및 전처리, UI/UX 설계 |
| **박영서** | 프론트엔드 개발, UI/UX 설계, 효창동 데이터 수집 및 전처리 |
| **이서아** | 프론트엔드 개발, 갈월동 데이터 수집 및 전처리 |
| **최서아** | 백엔드 개발, API 명세서 작성, 서버 관리, 남영동 및 학식 데이터 수집 |

---

## 프로젝트 개요

### 배경
현대인들은 하루에 평균 **30분**을 '무엇을 먹을까' 고민하는 데 소비합니다. 특히 알레르기, 지병, 특별한 식습관이 있는 사용자들은 적합한 메뉴를 찾기가 더욱 어렵습니다.

### 솔루션
**개인 맞춤형 AI 메뉴 추천 시스템**을 통해 사용자의 건강 정보와 취향을 분석하여 최적의 메뉴를 추천합니다. 귀여운 캐릭터 '쫩쫩이'와 함께 직관적이고 재미있는 사용자 경험을 제공합니다.

---

## 주요 기능

### 개인 맞춤 추천
- **건강 정보 기반**: 당뇨, 신장질환, 고혈압, 저혈압 등 지병 고려
- **알레르기 정보**: 달걀, 갑각류, 밀, 땅콩/대두, 고기, 콩, 우유 등
- **취향 반영**: 선호/비선호 메뉴 학습 및 적용

### 그룹 추천 (퀵픽)
- 여러 사용자의 정보를 종합하여 모두가 만족할 수 있는 메뉴 추천
- 공통 선호 메뉴 추출 및 위험 요소 자동 제거

### AI 챗봇 상황 추천
- 날씨, 시간, 기분 등을 고려한 자연어 기반 메뉴 추천
- OpenAI GPT API를 활용한 맥락적 추천

### 위치 기반 서비스
- GPS를 활용한 청파동, 효창동, 갈월동, 남영동 맛집 정보
- 거리별 정렬 및 접근성 정보 제공

### 피드백 학습
- 좋아요/싫어요 버튼을 통한 실시간 선호도 학습
- 별점 및 리뷰 태그 분석을 통한 추천 정확도 향상

---

## 추천 알고리즘

### 1. 컨텐츠 기반 필터링 (Content-Based Filtering)

```python
# TF-IDF 벡터화 및 코사인 유사도 계산
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(menu_df['combined_features'])
user_vector = vectorizer.transform([user_preference])
cos_sim = cosine_similarity(user_vector, tfidf_matrix)
```

**프로세스:**
1. 메뉴 이름, 식당 정보, 리뷰 태그 등을 조합한 텍스트 데이터 생성
2. TF-IDF로 벡터화 후 코사인 유사도 계산
3. 사용자 선호도와 가장 유사한 메뉴 추천

### 2. 사용자 정보 기반 필터링

```python
def filter_menus(user_info, menu_df):
    excluded = set(user_info['diseases'] + user_info['allergies'] + user_info['dislikes'])
    return menu_df[~menu_df['ingredients'].apply(
        lambda ing: any(item in ing for item in excluded)
    )]
```

- **제외 로직**: 지병, 알레르기, 비선호 메뉴 자동 필터링
- **가산점 시스템**: 선호 메뉴 포함 시 우선순위 상승

### 3. 그룹 추천 알고리즘

```python
def get_group_common_menus(group_profiles):
    all_sets = [set(p['preferred'] + p['neutral']) for p in group_profiles]
    return set.intersection(*all_sets)
```

### 4. 피드백 기반 학습

- **즉시 반영**: 좋아요/싫어요 피드백 실시간 적용
- **리뷰 분석**: 별점(1-5점) 및 긍정/부정 태그 가중치 반영

---

## 기술 스택

### Frontend
- **React.js** - 사용자 인터페이스
- **React Router** - 페이지 라우팅
- **Axios** - API 통신
- **CSS/Styled-components** - 스타일링

### Backend
- **FastAPI** - 웹 프레임워크
- **SQLAlchemy** - ORM
- **MySQL** - 데이터베이스
- **Uvicorn** - ASGI 서버

### AI/ML
- **Python** - 메인 언어
- **Scikit-learn** - 머신러닝 라이브러리
- **Pandas** - 데이터 처리
- **OpenAI API** - 자연어 처리

### Tools
- **Figma** - UI/UX 디자인
- **Git** - 버전 관리

---

## 시스템 아키텍처

```
┌─────────────────┐    HTTP/HTTPS    ┌─────────────────┐
│   React.js      │ ◄──────────────► │   FastAPI       │
│   Frontend      │                  │   Backend       │
└─────────────────┘                  └─────────────────┘
                                               │
                                               │ SQLAlchemy
                                               ▼
                            ┌─────────────────────────────────┐
                            │           MySQL DB              │
                            └─────────────────────────────────┘
                                               │
                                               │ Pandas + Sklearn
                                               ▼
                            ┌─────────────────────────────────┐
                            │     AI 추천 시스템              │
                            │  ┌─────────────────────────┐    │
                            │  │   OpenAI API           │    │
                            │  │   (상황별 추천)         │    │
                            │  └─────────────────────────┘    │
                            └─────────────────────────────────┘
```

---

## 주요 화면

### 메인 화면
- 쫩쫩이 캐릭터와 함께하는 직관적인 UI
- 개인 추천 / 그룹 추천 선택

### 프로필 설정
- 지병, 알레르기, 선호도 정보 입력
- 간편한 체크박스 형태의 입력 인터페이스

### 추천 결과
- 메뉴 이미지, 가격, 거리 정보 한눈에 보기
- 좋아요/싫어요 피드백 버튼

### 퀵픽 기능
- 친구 선택 및 그룹 맞춤 메뉴 추천
- 공통 선호 메뉴 시각화

### 리뷰 시스템
- 별점 평가 및 태그 선택
- 개인 식사 히스토리 관리

---

## 주요 API 엔드포인트

```python
# 메뉴 추천
@router.post("/api/menu-recommend")
def recommend_menu(user_input: UserInputModel):
    # 개인 맞춤 추천 로직
    pass

# 그룹 추천
@router.post("/api/quickpick")
def quick_pick(group_input: List[UserInputModel]):
    # 그룹 공통 메뉴 추출
    pass

# 피드백 처리
@router.post("/api/feedback")
def submit_feedback(feedback: FeedbackModel):
    # 사용자 피드백 학습
    pass

# AI 챗봇 추천
@router.post("/api/chatbot-recommend")
def chatbot_recommend(context: ContextModel):
    # OpenAI API 활용 상황별 추천
    pass
```

---

<div align="center">

**사용자 맞춤형 AI 메뉴 추천으로 더 나은 식사 경험을!**

Made with  by 숙명여자대학교 IPS 팀

</div>
