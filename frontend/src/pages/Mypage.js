import React, { useState, useEffect } from 'react';
import './Mypage.css';
import Modal from '../components/Modal';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useUserData } from '../UserDataContext';
import { API_BASE_URL } from '../api/api';

function Mypage() {
  const navigate = useNavigate();

  const {
    allergy, setAllergy,
    disease, setDisease,
    preferredMenu, setPreferredMenu,
    dislikedMenu, setDislikedMenu,
  } = useUserData();

  const username = localStorage.getItem('username');

  const allergyOptions = ["달걀", "갑각류", "밀", "땅콩/대두", "고기", "콩", "우유"];
  const diseaseOptions = ["고혈압", "저혈압", "당뇨", "신장질환"];
  const menuOptions = ["고기", "버섯", "고수", "내장", "닭발", "해산물"];

  const [modalInfo, setModalInfo] = useState({ isOpen: false, title: "", options: [], onSelect: () => {} });

  useEffect(() => {
    if (!username) return;

    axios.get(`${API_BASE_URL}/user/${username}`)
      .then(res => {
        const toArray = (value) =>
          typeof value === 'string'
            ? value.split(',').map(v => v.trim()).filter(Boolean)
            : Array.isArray(value)
            ? value
            : [];

        setAllergy(toArray(res.data.allergies));
        setDisease(toArray(res.data.diseases));
        setPreferredMenu(toArray(res.data.preferred_menu));
        setDislikedMenu(toArray(res.data.disliked_menu));
      })
      .catch(err => {
        console.error("유저 데이터 불러오기 실패:", err);
        setAllergy([]);
        setDisease([]);
        setPreferredMenu([]);
        setDislikedMenu([]);
      });
  }, [username, setAllergy, setDisease, setPreferredMenu, setDislikedMenu]);

  const removeItem = (list, setList, item) => {
    setList(list.filter(i => i !== item));
  };

  const openModal = (title, options, list, setList) => {
    const available = options.filter(o => !list.includes(o));
    if (available.length === 0) {
      alert("추가할 수 있는 항목이 없습니다.");
      return;
    }

    setModalInfo({
      isOpen: true,
      title,
      options: available,
      onSelect: (selected) => {
        setList([...list, selected]);
        closeModal();
      }
    });
  };

  const closeModal = () => {
    setModalInfo({ ...modalInfo, isOpen: false });
  };

  const handleSave = () => {
    if (!username || username.trim() === "") {
      alert("사용자 정보가 없습니다. 로그인이 필요합니다.");
      return;
    }

    const saveData = {
      username,
      allergies: Array.isArray(allergy) ? allergy.join(', ') : "",
      diseases: Array.isArray(disease) ? disease.join(', ') : "",
      preferred_menu: Array.isArray(preferredMenu) ? preferredMenu.join(', ') : "",
      disliked_menu: Array.isArray(dislikedMenu) ? dislikedMenu.join(', ') : ""
    };

    // ✅ localStorage에 userProfile 저장 (추천 페이지에서 사용)
    const userProfile = {
      user_id: Number(localStorage.getItem("user_id")) || 1,
      allergies: allergy,
      diseases: disease,
      preferred_menu: preferredMenu,
      disliked_menu: dislikedMenu
    };
    localStorage.setItem("userProfile", JSON.stringify(userProfile));

    axios.post(`${API_BASE_URL}/mypage/update`, saveData, {
      headers: {
        'Content-Type': 'application/json'
      }
    })
      .then(() => {
        alert("저장되었습니다!");
        setTimeout(() => {
          navigate("/eating-mode");
        }, 500);
      })
      .catch(err => {
        console.error("저장 오류 상세:", err.response?.data.detail || err);
        alert("저장 중 오류 발생! 콘솔을 확인해주세요.");
      });
  };

  const renderItem = (title, list, setList, options) => (
    <div className="mypage-item">
      <div className="mypage-box">
        <strong className="mypage-label">{title}</strong>
        <div className="mypage-tags">
          {list.map(item => (
            <span className="tag" key={item}>
              {item}
              <button className="remove-button" onClick={() => removeItem(list, setList, item)}>
                <img src={`/delete.png`} alt="삭제" className="delete-icon" />
              </button>
            </span>
          ))}
          <button className="add-button" onClick={() => openModal(`${title} 추가`, options, list, setList)}>＋</button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="mypage-container">
      <h1 className="mypage-title">마이페이지</h1>

      {renderItem("지병", disease, setDisease, diseaseOptions)}
      {renderItem("알러지", allergy, setAllergy, allergyOptions)}
      {renderItem("선호 메뉴", preferredMenu, setPreferredMenu, menuOptions)}
      {renderItem("비선호 메뉴", dislikedMenu, setDislikedMenu, menuOptions)}

      <div className="history-button-container">
        <button className="history-button" onClick={() => navigate("/history")}>
          히스토리 가기<span className="arrow">→</span>
        </button>
      </div>

      <div className="button-container">
        <button className="save-button" onClick={handleSave}>저장</button>
      </div>

      <Modal
        isOpen={modalInfo.isOpen}
        title={modalInfo.title}
        options={modalInfo.options}
        onSelect={modalInfo.onSelect}
        onClose={closeModal}
      />
    </div>
  );
}

export default Mypage;
