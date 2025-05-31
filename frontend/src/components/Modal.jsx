import React from 'react';
import './Modal.css';

function Modal({ isOpen, title, options, onSelect, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal-content">
        <h2>{title}</h2>
        <div className="modal-options">
          {options.map(option => (
            <button key={option} onClick={() => onSelect(option)}>
              {option}
            </button>
          ))}
        </div>
        <button className="modal-close" onClick={onClose}>닫기</button>
      </div>
    </div>
  );
}

export default Modal;
