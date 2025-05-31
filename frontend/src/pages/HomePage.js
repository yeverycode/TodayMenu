import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';
import axios from 'axios';
import { API_BASE_URL } from '../api/api';

function HomePage() {
  const navigate = useNavigate();

  const [nickname, setNickname] = useState("(닉네임)");
  const [location, setLocation] = useState("청파동"); // ✅ 기본값 고정

  const username = localStorage.getItem('username');

  // 닉네임 불러오기
  useEffect(() => {
    if (!username) return;

    axios.get(`${API_BASE_URL}/user/${username}`)
      .then(res => {
        setNickname(res.data.name || username);
      })
      .catch(err => {
        console.error("유저 정보 불러오기 실패:", err);
      });
  }, [username]);

  // 위치 가져오기 (※ 비활성화 처리)
  /*
  useEffect(() => {
    if (!navigator.geolocation) {
      setLocation("위치 정보를 지원하지 않습니다.");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;

        try {
          const response = await axios.get("https://dapi.kakao.com/v2/local/geo/coord2regioncode.json", {
            params: {
              x: longitude,
              y: latitude
            },
            headers: {
              Authorization: "KakaoAK 815a330dcfb69987a6c219836b68598c"
            }
          });

          if (response.data.documents.length > 0) {
            const regionName = response.data.documents[0].region_3depth_name;
            setLocation(regionName);
          } else {
            setLocation("지역 정보를 불러올 수 없습니다.");
          }
        } catch (error) {
          console.error("카카오 API 오류:", error);
          setLocation("지역 정보를 가져올 수 없습니다.");
        }
      },
      (error) => {
        console.error("위치 가져오기 오류:", error);
        setLocation("위치 정보를 가져올 수 없습니다.");
      }
    );
  }, []);
  */

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
      
      <button className="location-button">{location}</button>
      
      <button className="menu-recommend-button" onClick={() => navigate("/menu-recommend")}>
        메뉴 추천
      </button>

      <div className="home-buttons">
        <button className="home-sub-button chatbot" onClick={() => navigate("/chatbot")}>
          챗봇
        </button>
        <button className="home-sub-button quickpick" onClick={() => navigate("/quickpick-loading")}>
          퀵픽
        </button>
      </div>
    </div>
  );
}

export default HomePage;
