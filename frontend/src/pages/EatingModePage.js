import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './EatingModePage.css';
import axios from 'axios';
import { API_BASE_URL } from '../api/api'; // ⚠️ API 경로 정확히 확인

function EatingModePage() {
  const navigate = useNavigate();
  const [nickname, setNickname] = useState("(닉네임)");
  const username = localStorage.getItem("username");

  // 닉네임 불러오기
  useEffect(() => {
    if (!username) return;

    axios.get(`${API_BASE_URL}/user/${username}`)
      .then(res => {
        setNickname(res.data.name || username); // name이 없으면 username fallback
      })
      .catch(err => {
        console.error("닉네임 불러오기 실패:", err);
        setNickname(username); // 에러 시 fallback
      });
  }, [username]);

  return (
    <div className="page-container">
      <h1 className="page-title">
        <span className="word-first">오늘의&nbsp;</span> 
        <span className="word-middle"> 먹방</span>
        <span className="word-end">은</span>
      </h1>

      <div className="chef-container">
        <img src="/movetomypage.png" alt="Chef" className="home-image" />
      </div>

      <h2 className="nickname">{nickname}</h2>

      <button className="location-button">청파동</button>

      <div className="eating-choice-buttons">
        <button className="eating-button solo" onClick={() => navigate("/home")}>
          혼자 먹을래요
        </button>
        <button className="eating-button team" onClick={() => navigate("/team-pick")}>
          같이 먹을래요
        </button>
      </div>
    </div>
  );
}

export default EatingModePage;
