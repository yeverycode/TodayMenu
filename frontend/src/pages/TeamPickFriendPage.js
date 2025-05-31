import React, { useEffect, useState } from "react";
import "./TeamPickFriendPage.css";
import axios from "axios";
import { API_BASE_URL } from "../api/api";
import { useNavigate } from "react-router-dom";

function TeamPickFriendPage() {
  const [users, setUsers] = useState([]);
  const [selectedUsers, setSelectedUsers] = useState([]);
  const [searchKeyword, setSearchKeyword] = useState("");

  const navigate = useNavigate();

  useEffect(() => {
    if (searchKeyword.trim() === "") {
      setUsers([]);
      return;
    }

    const fetchUsers = async () => {
      try {
        const res = await axios.get(`${API_BASE_URL}/users/search`, {
          params: { keyword: searchKeyword },
        });
        setUsers(res.data);
      } catch (err) {
        console.error("유저 검색 실패:", err);
      }
    };

    const delayDebounce = setTimeout(fetchUsers, 300);
    return () => clearTimeout(delayDebounce);
  }, [searchKeyword]);

  const toggleSelect = (user) => {
    const exists = selectedUsers.find((u) => u.username === user.username);
    if (exists) {
      setSelectedUsers((prev) =>
        prev.filter((u) => u.username !== user.username)
      );
    } else {
      setSelectedUsers((prev) => [...prev, user]);
    }
  };

  const handleCreateGroup = () => {
    const usernames = selectedUsers.map((u) => u.username);
    localStorage.setItem("teamMembers", JSON.stringify(usernames));
    navigate("/team-loading", { state: { selected: usernames } });
  };

  const combinedUsers = [
    ...selectedUsers,
    ...users.filter(
      (user) => !selectedUsers.find((u) => u.username === user.username)
    ),
  ];

  return (
    <div className="friend-select-container">
      <div className="with-friend-title">
        <span
          style={{
            position: "absolute",
            left: 0,
            top: "50%",
            transform: "translateY(-50%)",
            cursor: "pointer",
            color: "#f4aab9",
            fontSize: "22px",
            fontFamily: "Hakgyoansim-Regular",
          }}
          onClick={() => navigate(-1)}
        >
          &lt;
        </span>
        <span className="pink-text">함께 먹을 친구&nbsp; </span>
        <span className="highlight">찾기</span>
      </div>

      <div className="search-wrapper">
        <input
          type="text"
          placeholder="친구 검색하기"
          className="search-bar"
          value={searchKeyword}
          onChange={(e) => setSearchKeyword(e.target.value)}
        />
        <img src="/search.png" alt="검색" className="search-icon" />
      </div>

      <ul className="user-list">
        {combinedUsers.map((user) => (
          <li
            key={user.id}
            className={`user-item ${
              selectedUsers.find((u) => u.username === user.username)
                ? "selected"
                : ""
            }`}
            onClick={() => toggleSelect(user)}
          >
            <img
              src={`/profile/${user.username}.png`}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "/movetomypage.png";
              }}
              alt={user.name || user.username}
              className="profile-img"
            />
            <span className="user-name">{user.name || user.username}</span>
            <div className="select-circle" />
          </li>
        ))}
      </ul>

      <div className="button-wrapper">
        <button className="create-group-btn" onClick={handleCreateGroup}>
          그룹 만들기
        </button>
      </div>
    </div>
  );
}

export default TeamPickFriendPage;
