import React, { useState, useEffect } from 'react';
import './DiseasePage.css';
import { useNavigate } from 'react-router-dom';
import { useUserData } from '../UserDataContext';
import axios from 'axios';
import { API_BASE_URL } from '../api/api';

function DiseasePage() {
  const navigate = useNavigate();
  const { setDisease } = useUserData();

  const diseases = ["고혈압", "저혈압", "당뇨", "신장질환"];
  const [selectedDiseases, setSelectedDiseases] = useState([]);

  useEffect(() => {
    const username = localStorage.getItem("username");
    if (!username) return;

    axios.get(`${API_BASE_URL}/user/${username}`)
      .then(res => {
        const existing = res.data.diseases;
        const toArray = (value) =>
          typeof value === 'string'
            ? value.split(',').map(v => v.trim()).filter(Boolean)
            : Array.isArray(value) ? value : [];
        setSelectedDiseases(toArray(existing));
      })
      .catch(err => {
        console.error("기존 지병 데이터 불러오기 실패:", err);
      });
  }, []);

  const toggleDisease = (item) => {
    setSelectedDiseases(prev =>
      prev.includes(item) ? prev.filter(d => d !== item) : [...prev, item]
    );
  };

  const handleNext = async () => {
    setDisease(selectedDiseases);
    const username = localStorage.getItem("username");

    try {
      const prev = await axios.get(`${API_BASE_URL}/user/${username}`);

      const cleanedDiseases = selectedDiseases.filter(Boolean).join(', ');
      const cleanedAllergies = Array.isArray(prev.data.allergies)
        ? prev.data.allergies.filter(Boolean).join(', ')
        : "";

      await axios.post(`${API_BASE_URL}/mypage/update`, {
        username,
        allergies: cleanedAllergies,
        diseases: cleanedDiseases,
        prefers: prev.data.prefers || [],
        dislikes: prev.data.dislikes || []
      }, {
        headers: { 'Content-Type': 'application/json' }
      });

      navigate("/preference");
    } catch (error) {
      console.error("지병 저장 실패:", error);
      alert("저장 중 오류 발생. 콘솔을 확인하세요.");
    }
  };

  return (
    <div className="disease-container">
      <h1 className="disease-title">
        <span className="pink">지병</span>
        <span className="brown"> 정보 입력</span>
      </h1>

      <p className="disease-description">
        앓고 있는 지병이 있으신가요?
      </p>

      <div className="disease-buttons">
        {diseases.map(item => (
          <button
            key={item}
            className={`disease-button ${selectedDiseases.includes(item) ? 'selected' : ''}`}
            onClick={() => toggleDisease(item)}
          >
            {item}
          </button>
        ))}
      </div>

      <div className="next-button-container">
        <button className="next-button" onClick={handleNext}>다음</button>
      </div>
    </div>
  );
}

export default DiseasePage;
