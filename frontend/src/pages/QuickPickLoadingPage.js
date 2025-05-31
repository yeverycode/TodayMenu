import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './QuickPickLoadingPage.css';

function QuickPickLoadingPage() {
  const navigate = useNavigate();

  useEffect(() => {
    const timer = setTimeout(() => {
      navigate('/quickpick-result');
    }, 3000);
    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <div className="quickpick-container">
      <img src="/quick_character.png" alt="Quickpick" className="quickpick-image" />
      <h1 className="quickpick-title"><span>퀵픽</span> 측정중...</h1>
      <p className="quickpick-description">
        열심히 입맛을 측정하고 있어요<br />
        과연 어떤 메뉴가 나올까요?
      </p>
    </div>
  );
}

export default QuickPickLoadingPage;
