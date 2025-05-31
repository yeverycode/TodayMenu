import React, { useState } from "react";
import axios from "axios";

function Recommendation() {
  const [allergies, setAllergies] = useState([]);
  const [weather, setWeather] = useState("");
  const [menus, setMenus] = useState([]);

  const allergyOptions = ["밀", "고기", "달걀", "땅콩/대두", "갑각류"];

  const toggleAllergy = (item) => {
    if (allergies.includes(item)) {
      setAllergies(allergies.filter(a => a !== item));
    } else {
      setAllergies([...allergies, item]);
    }
  };

  const getRecommendations = () => {
    axios.post("http://localhost:8000/recommend", {
      allergies: allergies,
      weather: weather
    }).then(res => {
      setMenus(res.data.recommendations);
    }).catch(err => {
      console.error(err);
    });
  };

  return (
    <div>
      <h2>알러지 선택</h2>
      {allergyOptions.map(item => (
        <button key={item} onClick={() => toggleAllergy(item)}>
          {allergies.includes(item) ? `[X] ${item}` : item}
        </button>
      ))}

      <h2>날씨 입력</h2>
      <input value={weather} onChange={(e) => setWeather(e.target.value)} placeholder="예: 흐림" />

      <br/><br/>
      <button onClick={getRecommendations}>추천 받기</button>

      <h2>추천 결과</h2>
      <ul>
        {menus.map((menu, idx) => (
          <li key={idx}>
            [{menu.place_name}] {menu.menu_name} - {menu.menu_price}원
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Recommendation;
