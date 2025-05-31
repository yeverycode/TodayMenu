import React, { useState, useEffect } from 'react';
import './MenuRecommendPage.css';
import { useNavigate } from 'react-router-dom';
import { API_BASE_URL } from '../api/api';
import { useUserData } from '../UserDataContext';

function MenuRecommendPage() {
  const [showFilters, setShowFilters] = useState(true);
  const [region, setRegion] = useState("청파동");
  const [alone, setAlone] = useState("혼자");
  const [budget, setBudget] = useState("1~2만원");
  const [drink, setDrink] = useState("없음");
  const [hunger, setHunger] = useState("보통");
  const [isLoading, setIsLoading] = useState(false);

  const navigate = useNavigate();
  const {
    allergy,
    disease,
    preferredMenu,
    dislikedMenu
  } = useUserData();

  useEffect(() => {
    const savedInfo = localStorage.getItem("recommendInfo");
    if (savedInfo) {
      const parsedInfo = JSON.parse(savedInfo);
      setRegion(parsedInfo.region || "청파동");
      setAlone(parsedInfo.alone || "혼자");
      setBudget(parsedInfo.budget || "1~2만원");
      setDrink(parsedInfo.drink || "없음");
      setHunger(parsedInfo.hunger || "보통");
    }
  }, []);

  const handleRecommend = async () => {
    setIsLoading(true);

    const userProfile = JSON.parse(localStorage.getItem("userProfile")) || {};
    const user_id = userProfile.user_id || 1;

    const allergies = allergy || [];
    const diseases = disease || [];
    const liked_ingredients = preferredMenu || [];
    const disliked_ingredients = dislikedMenu || [];

    const recommendInfo = {
      user_id,
      region,
      alone,
      budget,
      drink,
      hunger,
      allergies,
      diseases,
      liked_ingredients,
      disliked_ingredients
    };

    try {
      localStorage.setItem("recommendInfo", JSON.stringify(recommendInfo));

      const response = await fetch(`${API_BASE_URL}/api/menu-recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(recommendInfo),
      });

      if (!response.ok) throw new Error("서버 오류가 발생했습니다.");

      const result = await response.json();
      localStorage.setItem("recommendResult", JSON.stringify(result));
      navigate("/menu-result");
    } catch (error) {
      alert("메뉴 추천 중 오류가 발생했습니다. 다시 시도해주세요.");
      console.error("추천 요청 오류:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page-container">
      <h1 className="page-title">
        <span className="word-first">오늘의&nbsp;</span> <span className="word-middle">먹방</span><span className="word-end">은</span>
      </h1>

      <div className="options-section">
        <div className="option region-option">
          <div className="option-title">장소</div>
          <div className="select-wrapper">
            <select value={region} onChange={(e) => setRegion(e.target.value)} className="select-box">
              <option value="갈월동">갈월동</option>
              <option value="청파동">청파동</option>
              <option value="효창동">효창동</option>
              <option value="남영동">남영동</option>
              <option value="학식">교내 : 학식</option>
            </select>
            <img src="/select.png" alt="선택" className="select-icon" />
          </div>
        </div>

        {region === "학식" && (
          <p className="info-text">학식 메뉴는 예산/공복감 필터 없이 추천됩니다 🍱</p>
        )}

        {region !== "학식" && (
          <>
            <div className="option">
              <div className="option-title">예산</div>
              <div className="button-group">
                {["1만원 미만", "1~2만원", "2~3만원", "3~4만원", "4만원 이상"].map(item => (
                  <button key={item} className={budget === item ? "active" : ""} onClick={() => setBudget(item)}>{item}</button>
                ))}
              </div>
            </div>

            <div className="option additional-filters">
              <div className="filter-header" onClick={() => setShowFilters(!showFilters)}>
                <div className="option-title">추가 필터</div>
                <span className="toggle-icon">{showFilters ? "▲ 접기" : "▼ 펼치기"}</span>
              </div>

              {showFilters && (
                <div className="sub-filters">
                  <div className="sub-option">
                    <div className="sub-title">음주</div>
                    <div className="button-group">
                      {["없음", "소주", "맥주", "와인"].map(item => (
                        <button key={item} className={drink === item ? "active" : ""} onClick={() => setDrink(item)}>
                          <img src={`/${item}.png`} alt={item} className="drink-icon" />
                          {item}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="sub-option">
                    <div className="sub-title">공복감</div>
                    <div className="button-group">
                      {["적음", "보통", "많이"].map(item => (
                        <button key={item} className={hunger === item ? "active" : ""} onClick={() => setHunger(item)}>{item}</button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        <div className="recommend-button-section">
          <button className="recommend-button" onClick={handleRecommend} disabled={isLoading}>
            {isLoading ? "추천 중..." : "메뉴 추천 받기 🍽️"}
          </button>
        </div>
      </div>

      <div className="navigation-tabs">
        <button className="nav-tab" onClick={() => navigate("/eating-mode")}>
          <img src="/home.png" alt="홈" className="tab-icon" />
        </button>
        <button className="nav-tab" onClick={() => navigate("/chatbot")}>
          <img src="/movetomypage.png" alt="챗봇" className="tab-icon" />
        </button>
        <button className="nav-tab" onClick={() => navigate("/mypage")}>
          <img src="/mypage.png" alt="마이페이지" className="tab-icon" />
        </button>
      </div>
    </div>
  );
}

export default MenuRecommendPage;
