import React, { useState } from "react";
import axios from "axios";

function AiRecommend() {
  const [recommendedMenu, setRecommendedMenu] = useState(null);

  // 예시 유저 데이터
  const sampleUserData = [
    1, 0, 0, 0, 0,    // 알러지 (one-hot)
    0, 1, 0,          // 지병 (one-hot)
    1, 0, 0, 0,       // 선호 (one-hot)
    0, 1, 0, 0,       // 비선호 (one-hot)
    1, 0, 0, 0,       // 날씨 (one-hot)
    1,                // 혼밥 여부
    0.8               // 예산 (정규화)
  ];

  const getRecommendation = async () => {
    try {
      const response = await axios.post("http://localhost:8000/ai_recommend", {
        user_data: sampleUserData,
      });
      setRecommendedMenu(response.data.recommended_menu);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h2>AI 메뉴 추천</h2>
      <button onClick={getRecommendation}>추천 받기</button>
      {recommendedMenu && (
        <p style={{ fontSize: "24px", marginTop: "20px" }}>
          추천 메뉴: {recommendedMenu}
        </p>
      )}
    </div>
  );
}

export default AiRecommend;
