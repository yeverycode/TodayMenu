import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './TeamPickResultPage.css';
import MenuDetailModal from '../components/MenuDetailModal';
import axios from 'axios';
import { API_BASE_URL } from '../api/api';

const menus = [
  { name: "삼겹살", image: "/samgyeopsal.png", ingredients: ["삼겹살", "쌈장", "고기"] },
  { name: "버섯덮밥", image: "/beoseot_deopbap.png", ingredients: ["밥", "양송이버섯", "간장", "버섯"] },
  { name: "쌀국수", image: "/pho.png", ingredients: ["소고기", "숙주", "고수", "면", "밀"] },
  { name: "무뼈닭발", image: "/dakbal.png", ingredients: ["닭발", "고기", "고추장"] },
  { name: "해물찜", image: "/haemuljjim.png", ingredients: ["새우", "꽃게", "문어"] },
  { name: "스시", image: "/eggroll.png", ingredients: ["해산물"] },
  { name: "양꼬치", image: "/tofu.png", ingredients: ["양고기", "고기"] },
  { name: "크림파스타", image: "/cream_pasta.png", ingredients: ["면", "우유", "치즈", "크림"] },
  { name: "버거", image: "/burger.png", ingredients: ["빵", "고기", "치즈", "소스"] },
  { name: "치킨", image: "/chicken.png", ingredients: ["닭고기", "튀김옷", "고기", "밀"] },
  { name: "짜장면", image: "/jajangmyeon.png", ingredients: ["면", "춘장", "고기", "밀"] },
  { name: "떡볶이", image: "/tteokbokki.png", ingredients: ["떡", "고추장", "밀"] },
  { name: "마라탕", image: "/malatang.png", ingredients: ["고기", "버섯", "양고기", "고수"] },
  { name: "찜닭", image: "/jjimdak.png", ingredients: ["닭고기", "간장", "고기"] },
  { name: "육회비빔밥", image: "/yukhoe_bibimbap.png", ingredients: ["육회", "밥", "고기", "달걀"] },
  { name: "삼계탕", image: "/samgyetang.png", ingredients: ["닭고기", "인삼", "대추"] }, 
  { name: "쌈밥", image: "/ssambap.png", ingredients: ["상추", "쌈장", "밥", "고기"] }, // 고혈압
  { name: "샤브샤브", image: "/shabu.png", ingredients: ["채소", "소고기", "버섯"] }, // 고혈압
  { name: "유부초밥", image: "/yubuchobap.png", ingredients: ["유부", "밥", "식초"] }, // 신장질환
  { name: "감자전", image: "/gamjajeon.png", ingredients: ["감자", "소금"] }, // 신장질환
  { name: "콩국수", image: "/kongguksu.png", ingredients: ["콩", "면", "밀"] }, // 당뇨
  { name: "연어 샐러드", image: "/salmon_salad.png", ingredients: ["연어", "채소", "드레싱"] }, // 당뇨
  { name: "닭가슴살 샐러드", image: "/chicken_salad.png", ingredients: ["닭가슴살", "채소", "드레싱"] }, // 저혈압 & 신장
  { name: "버섯들깨탕", image: "/deulkkaetang.png", ingredients: ["버섯", "들깨", "채소"] }, // 고혈압 & 신장
  { name: "호박죽", image: "/hobakjuk.png", ingredients: ["단호박", "쌀", "우유"] }, // 저혈압 & 당뇨
  { name: "두부 샐러드", image: "/tofu_salad.png", ingredients: ["두부", "채소", "드레싱"] }, // 고혈압 & 당뇨
  { name: "연어 구이", image: "/grilled_salmon.png", ingredients: ["연어", "소금", "올리브오일"] } // 고혈압 & 당뇨
];

const INGREDIENT_CATEGORIES = {
  "고기": ["소고기", "돼지고기", "닭고기", "삼겹살", "양고기", "고기", "육회", "닭발", "닭가슴살"],
  "해산물": ["새우", "꽃게", "문어", "연어", "해산물"],
  "채소": ["채소", "상추", "숙주", "고수", "감자", "단호박", "인삼", "대추", "버섯", "양송이버섯", "표고버섯"],
  "탄수화물": ["밥", "떡", "쌀", "면", "밀", "빵", "유부", "토스트", "감자"],
  "유제품": ["우유", "치즈", "크림", "생크림", "라떼"],
  "계란": ["달걀", "계란"],
  "가공식품": ["두부", "유부", "소스", "드레싱", "쌈장", "춘장", "간장", "고추장", "식초", "들깨", "올리브오일"],
  "조미료/소스": ["간장", "쌈장", "춘장", "고추장", "식초", "소금", "올리브오일"],
  "기타": ["콩", "들깨", "식초", "드레싱", "유부", "쌈장"]
};


const DISEASE_RULES = {
  "고혈압": ["간장", "된장", "햄", "소시지", "소금", "고기"],
  "저혈압": ["커피", "콜라", "에너지음료"],
  "당뇨": ["설탕", "시럽", "크림", "케이크", "빵"],
  "신장질환": ["된장", "소금", "소시지", "햄", "고기", "치즈"]
};

function normalize(str) {
  return str.trim().toLowerCase();
}

function safeArray(input) {
  if (Array.isArray(input)) return input;
  if (typeof input === 'string') return input.split(',').map(s => s.trim());
  return [];
}

function categorizeIngredient(ingredient, preferences) {
  const normalized = normalize(ingredient);
  const normalizedAllergies = safeArray(preferences.allergy).map(normalize);
  const normalizedDiseases = safeArray(preferences.disease).map(normalize);
  const normalizedLikes = safeArray(preferences.likes).map(normalize);
  const normalizedDislikes = safeArray(preferences.dislikes).map(normalize);

  let matchedCategory = null;

  for (const category in INGREDIENT_CATEGORIES) {
    const words = INGREDIENT_CATEGORIES[category].map(normalize);
    if (words.includes(normalized)) {
      matchedCategory = category;
      break;
    }
  }

  const isAllergy =
    normalizedAllergies.includes(normalized) ||
    (matchedCategory && normalizedAllergies.includes(normalize(matchedCategory)));

  const restrictedBy = normalizedDiseases.filter(disease =>
    (DISEASE_RULES[disease] || []).map(normalize).includes(normalized)
  );

  const isDislike =
    normalizedDislikes.includes(normalized) ||
    (matchedCategory && normalizedDislikes.includes(normalize(matchedCategory)));

  const isLike =
    normalizedLikes.includes(normalized) ||
    (matchedCategory && normalizedLikes.includes(normalize(matchedCategory)));

  return {
    category: matchedCategory,
    like: isLike,
    dislike: isDislike,
    allergy: isAllergy,
    restrictedBy
  };
}

function getReason(menu, preferences) {
  const liked = new Set();
  const disliked = new Set();
  const allergic = new Set();
  const restricted = new Map();

  menu.ingredients.forEach(ing => {
    const analysis = categorizeIngredient(ing, preferences);
    if (analysis.like) liked.add(ing);
    if (analysis.dislike) disliked.add(ing);
    if (analysis.allergy) allergic.add(ing);
    analysis.restrictedBy.forEach(disease => {
      restricted.set(ing, disease);
    });
  });

  return {
    liked: Array.from(liked),
    disliked: Array.from(disliked),
    allergic: Array.from(allergic),
    restricted: Array.from(restricted.entries())
  };
}


function isMenuDisqualified(menu, teamPreferences) {
  for (const prefs of teamPreferences) {
    for (const ingredient of menu.ingredients) {
      const result = categorizeIngredient(ingredient, prefs);
      if (
        result.restrictedBy.length > 0 ||
        result.allergy ||
        result.dislike
      ) {
        return true; // 하나라도 해당되면 탈락
      }
    }
  }
  return false; // 모두 통과 시만 true
}


function calculateTeamPreference(menu, teamPreferences) {
  if (isMenuDisqualified(menu, teamPreferences)) return 0;
  let totalScore = 0;
  for (const prefs of teamPreferences) {
    let score = 50, bonus = 0, penalty = 0;
    for (const ingredient of menu.ingredients) {
      const result = categorizeIngredient(ingredient, prefs);
      if (result.like) bonus += 15;
      if (result.dislike) penalty += 10;
    }
    score += bonus - penalty;
    score = Math.max(30, Math.min(score, 90));
    totalScore += score;
  }
  const avgScore = totalScore / teamPreferences.length;
  return Math.round(Math.max(0, Math.min(avgScore + (Math.random() * 6 - 3), 100)));
}

function getQuickPickMenus(teamPreferences) {
  const preferenceSets = teamPreferences.map(p => new Set([
    ...safeArray(p.likedMenu),
    ...safeArray(p.normalMenu)
  ]));

  const commonMenus = [...preferenceSets[0]].filter(menu =>
    preferenceSets.every(set => set.has(menu))
  );

  const safeMenus = commonMenus.filter(name => {
    const menu = menus.find(m => m.name === name);
    if (!menu) return false;
    return !teamPreferences.some(prefs =>
      menu.ingredients.some(ing => {
        const result = categorizeIngredient(ing, prefs);
        return result.allergy || result.restrictedBy.length > 0 || result.dislike;
      })
    );
  });

  return menus.filter(m => safeMenus.includes(m.name));
}

function TeamPickResultPage() {
  const [teamPreferences, setTeamPreferences] = useState([]);
  const [menuCandidates, setMenuCandidates] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [bestMenu, setBestMenu] = useState(null);
  const [percentage, setPercentage] = useState(0);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalInfo, setModalInfo] = useState({});
  const [isQuickPick, setIsQuickPick] = useState(false);
  const navigate = useNavigate();
  const teamMembers = JSON.parse(localStorage.getItem("teamMembers")) || [];

  useEffect(() => {
    async function fetchTeamPreferences() {
      try {
        const responses = await Promise.all(
          teamMembers.map(name => axios.get(`${API_BASE_URL}/mypage/${name}`))
        );
        const preferences = responses.map(res => res.data.preferences);
        setTeamPreferences(preferences);
      } catch (err) {
        console.error("❌ 팀 선호 정보 불러오기 실패:", err);
      }
    }
    fetchTeamPreferences();
  }, []);

  useEffect(() => {
    if (teamPreferences.length === 0) return;
    const candidates = (isQuickPick ? getQuickPickMenus(teamPreferences) : menus)
      .map(menu => ({
        ...menu,
        score: calculateTeamPreference(menu, teamPreferences),
        disqualified: isMenuDisqualified(menu, teamPreferences)
      }))
      .filter(menu => !menu.disqualified)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3);
    setMenuCandidates(candidates);
    setCurrentIndex(0);
  }, [teamPreferences, isQuickPick]);

  useEffect(() => {
    if (menuCandidates.length === 0) return;
    const selected = menuCandidates[currentIndex];
    if (!selected) return;
    setBestMenu(selected);
    setPercentage(0);
    const interval = setInterval(() => {
      setPercentage(prev => {
        if (prev < selected.score) return prev + 1;
        clearInterval(interval);
        return selected.score;
      });
    }, 20);
    return () => clearInterval(interval);
  }, [menuCandidates, currentIndex]);

  const viewDetail = () => {
    const reason = teamPreferences.map((prefs, idx) => ({
      member: teamMembers[idx],
      reason: getReason(bestMenu, prefs)
    }));
    setModalInfo({ menuName: bestMenu.name, percentage, reason });
    setIsModalOpen(true);
  };

  return (
    <div className="page-container">
      <h2 className="result-title">
        <span className="brown">그룹 메뉴 추천</span>
        <span className="pink"> 완료!</span>
      </h2>

      {bestMenu ? (
        <>
          <div className="result-circle">
            <div className="circle-fill" style={{ height: `${percentage}%` }}></div>
            <div className="circle-inner">
              <img src={bestMenu.image} alt="메뉴" className="menu-image" />
              <h2>{bestMenu.name}</h2>
              <p>{percentage}%</p>
            </div>
          </div>
          <p className="description">
            그룹의 선호가<br />
            <span className="percentage">{percentage}%</span> 반영된 최적의 메뉴에요!
            <span className="detail-link" onClick={viewDetail}>추천 상세 보기 &gt;</span>
          </p>
        </>
      ) : (
        <p className="description">😢 조건에 맞는 공통 메뉴가 없습니다.</p>
      )}

      <div className="result-buttons">
        <button
          className="common-button other-menu"
          onClick={() => setCurrentIndex((prev) => (prev + 1) % menuCandidates.length)}
        >
          다른 메뉴
        </button>
        <button className="common-button main-button" onClick={() => navigate('/home')}>
          메인으로
        </button>
      </div>

      <MenuDetailModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        menuName={modalInfo.menuName}
        percentage={modalInfo.percentage}
        reason={modalInfo.reason}
      />
    </div>
  );
}

export default TeamPickResultPage;