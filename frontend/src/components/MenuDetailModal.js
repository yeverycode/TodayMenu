import React from 'react';
import './MenuDetailModal.css';

function MenuDetailModal({ isOpen, onClose, menuName, percentage, reason }) {
  if (!isOpen) return null;

  const renderReasonList = (r) => (
    <ul>
      {r.liked?.length > 0 && (
        <li><strong>좋아하는 재료:</strong> {r.liked.join(", ")}</li>
      )}
      {r.disliked?.length > 0 && (
        <li><strong>싫어하는 재료:</strong> {r.disliked.join(", ")}</li>
      )}
      {r.allergic?.length > 0 && (
        <li style={{ color: "red" }}>
          ⚠️ <strong>알레르기 주의:</strong> {r.allergic.join(", ")}
        </li>
      )}
      {r.restricted?.length > 0 && (
        <li style={{ color: "orange" }}>
          💡 <strong>지병 제한:</strong>{" "}
          {r.restricted.map(([ing, disease], i) => (
            <span key={i}>
              {ing}({disease}){i < r.restricted.length - 1 ? ', ' : ''}
            </span>
          ))}
        </li>
      )}
      {r.liked?.length === 0 && r.disliked?.length === 0 && r.allergic?.length === 0 && r.restricted?.length === 0 && (
        <li><em>선호, 비선호 재료가 없어 기본 점수가 부여되었어요.</em></li>
      )}
    </ul>
  );

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>추천 결과 안내</h2>
        <p><strong>추천 메뉴:</strong> {menuName}</p>
        <p><strong>추천 점수:</strong> {percentage}%</p>

        {/* 단일 사용자 추천 사유 */}
        {reason && typeof reason === 'object' && !Array.isArray(reason) && (
          renderReasonList(reason)
        )}

        {/* 팀 기반 추천 사유 */}
        {Array.isArray(reason) && reason.map((item, idx) => (
          <div key={idx} className="reason-box">
            <h4>👤 {item.member} 님</h4>
            {renderReasonList(item.reason)}
          </div>
        ))}

        <button onClick={onClose} className="close-button">확인했어요</button>
      </div>
    </div>
  );
}

export default MenuDetailModal;
