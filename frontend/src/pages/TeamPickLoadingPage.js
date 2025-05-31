import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './TeamPickLoadingPage.css';

function TeamPickLoadingPage() {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigate('/team-result');
    }, 3000);
    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <div className="teampick-container">
      <img src="/quick_character.png" alt="TeamPick" className="teampick-image" />
      <h1 className="teampick-title"><span>그룹메뉴</span> 찾는중...</h1>
      <p className="teampick-description">
        열심히 그룹의 선호를 측정하고 있어요<br />
        과연 어떤 메뉴가 나올까요?
      </p>
    </div>
  );
}

export default TeamPickLoadingPage;
