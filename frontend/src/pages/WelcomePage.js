import React from 'react';
import { useNavigate } from 'react-router-dom';
import './WelcomePage.css';

function WelcomePage() {
  const navigate = useNavigate();

  return (
    <div className="container">
      <img src="/chef.png" alt="오늘의 먹방은 캐릭터" className="character" />
      <h1 class="title">
        <span class="pink">오늘의</span>
        <span class="brown"> 먹방</span>
        <span class="pink">은</span>
      </h1>

      <button className="start-button" onClick={() => navigate("/register")}>
        시작하기
      </button>
      <p className="login-text">
        이미 계정이 있나요?{" "}
        <span className="login-link" onClick={() => navigate("/login")}>
          로그인
        </span>
      </p>
    </div>
  );
}

export default WelcomePage;
