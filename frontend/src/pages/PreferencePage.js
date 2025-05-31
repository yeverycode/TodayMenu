import React, { useState } from 'react';
import './PreferencePage.css';
import { useNavigate } from 'react-router-dom';
import { useUserData } from '../UserDataContext';
import axios from 'axios';
import { API_BASE_URL } from '../api/api';

function PreferencePage() {
  const navigate = useNavigate();
  const {
    setPreferredMenu,
    setDislikedMenu
  } = useUserData();

  const items = [
    { name: "고기", image: "/고기.png" },
    { name: "버섯", image: "/버섯.png" },
    { name: "고수", image: "/고수.png" },
    { name: "내장", image: "/내장.png" },
    { name: "닭발", image: "/닭발.png" },
    { name: "해산물", image: "/해산물.png" }
  ];

  const [preferences, setPreferences] = useState({});

  const handlePreferenceChange = (itemName, value) => {
    setPreferences(prev => ({ ...prev, [itemName]: value }));
  };

  const handleSave = async () => {
    const liked = [];
    const disliked = [];

    Object.entries(preferences).forEach(([key, value]) => {
      if (value === "좋아요") liked.push(key);
      if (value === "싫어요") disliked.push(key);
    });

    const username = localStorage.getItem("username");

    try {
      const prev = await axios.get(`${API_BASE_URL}/user/${username}`);

      const allergies = Array.isArray(prev.data.allergies)
        ? prev.data.allergies.filter(Boolean).join(', ')
        : typeof prev.data.allergies === 'string'
        ? prev.data.allergies
        : "";

      const diseases = Array.isArray(prev.data.diseases)
        ? prev.data.diseases.filter(Boolean).join(', ')
        : typeof prev.data.diseases === 'string'
        ? prev.data.diseases
        : "";

      await axios.post(`${API_BASE_URL}/mypage/update`, {
        username,
        allergies,
        diseases,
        preferred_menu: liked.join(', '),
        disliked_menu: disliked.join(', ')
      }, {
        headers: { 'Content-Type': 'application/json' }
      });

      const updated = await axios.get(`${API_BASE_URL}/user/${username}`);

      const toArray = (value) =>
        typeof value === 'string'
          ? value.split(',').map(v => v.trim()).filter(Boolean)
          : Array.isArray(value)
          ? value
          : [];

      setPreferredMenu(toArray(updated.data.preferred_menu));
      setDislikedMenu(toArray(updated.data.disliked_menu));

      navigate("/mypage");
    } catch (error) {
      console.error("서버 저장 오류:", error);
      alert("저장 중 오류가 발생했습니다. 콘솔을 확인해주세요.");
    }
  };

  return (
    <div className="preference-container">
      <h1 className="preference-title">
        <span className="pink">음식 </span>
        <span className="brown">취향을 알려주세요!</span>
      </h1>

      <div className="preference-list">
        {items.map(item => (
          <div className="preference-item" key={item.name}>
            <img src={item.image} alt={item.name} className="preference-image" />
            <div className="preference-content">
              <div className="preference-name">{item.name}</div>
              <div className="preference-options">
                {["좋아요", "보통", "싫어요"].map(option => (
                  <label
                    key={option}
                    className={`preference-option ${preferences[item.name] === option ? 'selected' : ''}`}
                    onClick={() => handlePreferenceChange(item.name, option)}
                  >
                    <input type="radio" name={item.name} value={option} />
                    {option}
                  </label>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="next-button-container">
        <button className="next-button" onClick={handleSave}>저장</button>
      </div>
    </div>
  );
}

export default PreferencePage;
