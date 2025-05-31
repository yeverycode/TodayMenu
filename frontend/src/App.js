import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import WelcomePage from './pages/WelcomePage';
import RegisterPage from './pages/RegisterPage';
import AllergyPage from './pages/AllergyPage';
import DiseasePage from './pages/DiseasePage';
import PreferencePage from './pages/PreferencePage';
import Mypage from './pages/Mypage';
import HomePage from './pages/HomePage';
import MenuRecommendPage from './pages/MenuRecommendPage';
import QuickPickLoadingPage from './pages/QuickPickLoadingPage';
import QuickPickResultPage from './pages/QuickPickResultPage';
import MenuResultPage from './pages/MenuResultPage';
import AiRecommendPage from './pages/AiRecommendPage';
import LoginPage from './pages/LoginPage';
import { UserDataProvider } from './UserDataContext';
import ChatBotPage from './pages/ChatBotPage';
import ReviewPage from './pages/ReviewPage';
import TeamPickFriendPage from './pages/TeamPickFriendPage'; 
import TeamPickLoadingPage from './pages/TeamPickLoadingPage';
import TeamPickResultPage from './pages/TeamPickResultPage';
import EatingModePage from './pages/EatingModePage';
import HistoryPage from './pages/HistoryPage';


function App() {
  return (
    <UserDataProvider>
      <Router>
        <Routes>
          <Route path="/" element={<WelcomePage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/allergy" element={<AllergyPage />} />
          <Route path="/disease" element={<DiseasePage />} />
          <Route path="/preference" element={<PreferencePage />} />
          <Route path="/mypage" element={<Mypage />} />
          <Route path="/home" element={<HomePage />} />
          <Route path="/menu-recommend" element={<MenuRecommendPage />} />
          <Route path="/quickpick-loading" element={<QuickPickLoadingPage />} />
          <Route path="/quickpick-result" element={<QuickPickResultPage />} />
          <Route path="/menu-result" element={<MenuResultPage />} />
          <Route path="/ai-recommend" element={<AiRecommendPage />} />
          <Route path="/chatbot" element={<ChatBotPage />} />
          <Route path="/review" element={<ReviewPage />} />
          <Route path="/team-pick" element={<TeamPickFriendPage />} />
          <Route path="/team-loading" element={<TeamPickLoadingPage />} />
          <Route path="/team-result" element={<TeamPickResultPage />} />
          <Route path="/eating-mode" element={<EatingModePage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </Router>
    </UserDataProvider>
  );
}

export default App;
