import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { API_BASE_URL } from "../api/api";
import axios from "axios";
import "./ReviewPage.css";

function ReviewPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { place, menu, image, restaurant_id, menu_id } = location.state || {};

  const [rating, setRating] = useState(0);
  const [tags, setTags] = useState([]);
  const [customTag, setCustomTag] = useState("");
  const [comment, setComment] = useState("");
  const [imageSrc, setImageSrc] = useState(null);
  const [imgTryIndex, setImgTryIndex] = useState(0);
  const [isAlreadyReviewed, setIsAlreadyReviewed] = useState(false);

  const tagOptions = ["좋아요", "별로예요", "가성비가 좋아요", "빨리 나와요", "접근성이 좋아요"];

  const toFileName = (text) => text?.trim().replace(/\s+/g, "_") || "";

  useEffect(() => {
    if (!restaurant_id || !menu_id) {
      alert("리뷰 작성에 필요한 정보가 없습니다. 다시 추천을 받아주세요.");
      navigate("/home");
      return;
    }

    const username = localStorage.getItem("username");
    if (!username) return;

    if (place && menu) {
      const baseName = `${toFileName(place)}_${toFileName(menu)}`;
      setImageSrc(`/menu-images/${baseName}.jpg`);
      setImgTryIndex(0);
    }

    axios
      .get(`${API_BASE_URL}/api/review/check`, {
        params: { username, restaurant_id, menu_id },
      })
      .then((res) => {
        if (res.data?.exists) {
          const review = res.data.review;
          setRating(review.rating);
          setTags(review.tags || []);
          setComment(review.comment || "");
          setIsAlreadyReviewed(true);
        }
      })
      .catch((err) => {
        console.error("리뷰 확인 실패:", err);
      });
  }, [place, menu, restaurant_id, menu_id, navigate]);

  const handleImageError = () => {
    const baseName = `${toFileName(place)}_${toFileName(menu)}`;
    const extensions = [".jpeg", ".png"];
    const nextIndex = imgTryIndex;

    if (nextIndex < extensions.length) {
      setImageSrc(`/menu-images/${baseName}${extensions[nextIndex]}`);
      setImgTryIndex(nextIndex + 1);
    } else {
      setImageSrc("/menu-images/기타.jpg");
    }
  };

  const handleTagClick = (tag) => {
    if (isAlreadyReviewed) return;
    setTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  const handleSubmit = async () => {
    const username = localStorage.getItem("username");
    const userId = localStorage.getItem("user_id");

    if (!username || !userId) {
      alert("로그인이 필요합니다.");
      navigate("/login");
      return;
    }

    if (rating === 0) {
      alert("별점을 선택해주세요.");
      return;
    }

    const reviewData = {
      username,
      restaurant_id,
      menu_id,
      rating: Number(rating),
      tags: tags.concat(customTag ? [customTag] : []),
      comment,
    };

    const feedbackData = {
      place_name: place,
      menu_name: menu,
      feedback: rating >= 4 ? "good" : rating <= 2 ? "bad" : "neutral",
      user_id: parseInt(userId),
    };

    console.log("✅ 보낼 reviewData:", reviewData);
    console.log("✅ 보낼 feedbackData:", feedbackData);

    try {
      await axios.post(`${API_BASE_URL}/api/review`, reviewData);

      // if (feedbackData.feedback !== "neutral") {
      //   await axios.post(`${API_BASE_URL}/api/feedback`, feedbackData);
      // }

      alert("리뷰가 저장되었습니다!");
      navigate("/history");
    } catch (err) {
      console.error("❌ 리뷰 저장 오류:", err.response || err);
      const message = err.response?.data?.detail || "서버 오류가 발생했습니다.";
      alert("리뷰 저장 실패: " + message);
    }
  };

  return (
    <div className="review-container">
      <img src="/chef.png" alt="캐릭터" className="review-character" />
      <h2 className="review-title">
        {isAlreadyReviewed ? "이미 작성한 리뷰입니다" : "리뷰를 작성해주세요"}
      </h2>

      <div className="review-place">
        {imageSrc && (
          <img
            src={imageSrc}
            alt={menu}
            className="menu-image"
            onError={handleImageError}
          />
        )}

        <h3 className="place-name">{place}</h3>
        <p className="menu-name">{menu}</p>

        <div className="stars">
          {[1, 2, 3, 4, 5].map((star) => (
            <img
              key={star}
              src={star <= rating ? "/filled_heart.png" : "/blank_heart.png"}
              alt={`${star}점`}
              className="heart-image"
              onClick={() => !isAlreadyReviewed && setRating(star)}
            />
          ))}
        </div>

        <div className="tags">
          {tagOptions.map((tag) => (
            <span
              key={tag}
              className={tags.includes(tag) ? "selected" : ""}
              onClick={() => handleTagClick(tag)}
            >
              {tag}
            </span>
          ))}
          <input
            type="text"
            className="custom-tag"
            placeholder="+"
            value={customTag}
            onChange={(e) => !isAlreadyReviewed && setCustomTag(e.target.value)}
            disabled={isAlreadyReviewed}
          />
        </div>

        <textarea
          className="comment-box"
          placeholder="리뷰 내용을 입력하세요. 한 번 작성한 리뷰는 수정이 불가합니다."
          value={comment}
          onChange={(e) => !isAlreadyReviewed && setComment(e.target.value)}
          disabled={isAlreadyReviewed}
        />

        {!isAlreadyReviewed && (
          <button className="save-button" onClick={handleSubmit}>
            저장
          </button>
        )}
      </div>
    </div>
  );
}

export default ReviewPage;
