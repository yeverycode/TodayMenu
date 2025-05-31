import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import "./ChatBotPage.css";
import { API_BASE_URL } from "../api/api";

function ChatBotPage() {
  const [messages, setMessages] = useState([
    { type: "bot", text: "안녕하세요, 쫩쫩이입니다.\\n어떤 메뉴를 추천해드릴까요?" }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [weather, setWeather] = useState("날씨 정보 없음");

  const navigate = useNavigate();
  const userId = localStorage.getItem("user_id") || "1"; // 기본값 "1" 테스트용

  const getWeatherDescription = async () => {
    try {
      if (!navigator.geolocation) return "날씨 정보 없음";

      const position = await new Promise((resolve, reject) =>
        navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 10000 })
      );
      const { latitude, longitude } = position.coords;

      const API_KEY = "dd16815f42e8f773eb808f36aeab2a31";
      const url = `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&appid=${API_KEY}&lang=kr`;
      const res = await fetch(url);
      if (!res.ok) return "날씨 정보 없음";

      const data = await res.json();
      const main = data.weather?.[0]?.main;
      const weatherMap = {
        Clear: "맑음",
        Clouds: "흐림",
        Rain: "비오는 날",
        Drizzle: "이슬비",
        Thunderstorm: "천둥 번개",
        Snow: "눈 오는 날",
        Mist: "안개",
        Haze: "연무"
      };
      return weatherMap[main] || "날씨 정보 없음";
    } catch (error) {
      console.error("날씨 감지 오류:", error);
      return "날씨 정보 없음";
    }
  };

  useEffect(() => {
    getWeatherDescription().then(setWeather);
  }, []);

  const sendMessage = () => {
    if (!input.trim()) return;

    setMessages((prev) => [...prev, { type: "user", text: input }]);
    setInput("");
    setIsLoading(true);

    const eventSource = new EventSource(
      `${API_BASE_URL}/api/llm-recommend-stream?user_id=${userId}&weather=${encodeURIComponent(weather)}&situation=${encodeURIComponent(input)}`
    );

    let botResponse = "";

    eventSource.onmessage = (event) => {
      if (event.data === "[END]") {
        const finalResponse = botResponse.trim();
        setMessages((prev) => [
          ...prev,
          {
            type: "bot",
            text: finalResponse || "추천 결과가 비어있어요. 다시 시도해 주세요."
          }
        ]);
        setIsLoading(false);
        eventSource.close();
      } else {
        botResponse += event.data;
      }
    };

    eventSource.onerror = (error) => {
      console.error("Stream error:", error);
      eventSource.close();
      setIsLoading(false);
      setMessages((prev) => [
        ...prev,
        { type: "bot", text: "추천 중 오류가 발생했어요. 다시 시도해 주세요." }
      ]);
    };
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>
          <span className="text-pink">오늘의</span>
          <span className="text-brown"> 먹방</span>
          <span className="text-pink">은</span>
        </h1>
      </div>

      <div className="chat-box">
        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.type}`}>
              {msg.type === "bot" && (
                <div className="chat-profile">
                  <div className="profile-wrapper">
                    <img src="/chatbot.png" alt="쫩쫩이" className="profile-img" />
                  </div>
                </div>
              )}
              <div className="message-content">
                <ReactMarkdown
                  breaks={true}
                  components={{
                    a: ({ node, ...props }) => (
                      <a {...props} style={{ color: "#c05d00" }} target="_blank" rel="noreferrer">
                        {props.children}
                      </a>
                    )
                  }}
                >
                  {msg.text.replaceAll("\\n", "\n")}
                </ReactMarkdown>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="message bot">
              <div className="chat-profile">
                <div className="profile-wrapper">
                  <img src="/chatbot.png" alt="쫩쫩이" className="profile-img" />
                </div>
              </div>
              <div className="message-content">추천하는 중입니다... 🍜</div>
            </div>
          )}
        </div>

        <div className="chat-input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="눌러서 쫩쫩이에게 메뉴 추천받기"
            className="chat-input"
          />
          <button onClick={sendMessage} className="send-button">
            <img src="/send.png" alt="보내기" className="send-icon" />
          </button>
        </div>
      </div>

      <div className="navigation-tabs">
        <button className="nav-tab" onClick={() => navigate("/eating-mode")}>
          <img src="/home.png" alt="홈" className="tab-icon" />
        </button>
        <button className="nav-tab" onClick={() => navigate("/chatbot")}>
          <img src="/movetomypage.png" alt="챗봇" className="tab-icon" />
        </button>
        <button className="nav-tab" onClick={() => navigate("/mypage")}>
          <img src="/mypage.png" alt="마이페이지" className="tab-icon" />
        </button>
      </div>
    </div>
  );
}

export default ChatBotPage;
