import React, { useState } from 'react';
import './RegisterPage.css';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { API_BASE_URL } from '../api/api';

function RegisterPage() {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    username: '',
    password: '',
    confirmPassword: '',
    name: '',
    phone: '',
    email: ''
  });

  const [errors, setErrors] = useState({});

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const validatePhone = (phone) => /^01[0-9]\d{7,8}$/.test(phone);
  const validateEmail = (email) => /^[\w.-]+@[\w.-]+\.\w{2,}$/.test(email);

  const handleSubmit = async () => {
    const newErrors = {};
    if (!formData.username) newErrors.username = '아이디를 입력해주세요';
    if (!formData.password) newErrors.password = '비밀번호를 입력해주세요';
    if (formData.password !== formData.confirmPassword)
      newErrors.confirmPassword = '비밀번호가 일치하지 않습니다';
    if (!formData.name) newErrors.name = '이름을 입력해주세요';
    if (formData.phone && !validatePhone(formData.phone)) newErrors.phone = '휴대폰 번호 형식이 올바르지 않습니다';
    if (formData.email && !validateEmail(formData.email)) newErrors.email = '이메일 형식이 올바르지 않습니다';

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    try {
      await axios.post(`${API_BASE_URL}/register`, {
          username: formData.username,
          password: formData.password,
          confirm_password: formData.confirmPassword, // ✅ 이 줄 추가
          name: formData.name,
          phone: formData.phone,
          email: formData.email
      });

      await axios.post(`${API_BASE_URL}/mypage/update`, {
        username: formData.username,
        allergies: "",
        diseases: "",
        preferred_menu: "",
        disliked_menu: ""
      });

      localStorage.setItem('username', formData.username);
      navigate('/allergy');
    } catch (error) {
      console.error('회원가입 오류:', error);

      if (error.response && error.response.data) {
        const detail = error.response.data.detail;

        if (
          detail === 'Username already exists' ||
          detail === 'Username already registered'
        ) {
          setErrors({ ...errors, username: '이미 사용 중인 아이디입니다' });
        } else if (typeof detail === 'string') {
          alert('회원가입 중 오류가 발생했습니다: ' + detail);
        } else if (Array.isArray(detail)) {
          const fieldError = detail[0];
          const loc = fieldError?.loc?.join('.') || '필드';
          const msg = fieldError?.msg || '오류가 발생했습니다.';
          alert(`회원가입 중 오류가 발생했습니다: ${loc} - ${msg}`);
        } else {
          alert('회원가입 중 알 수 없는 오류가 발생했습니다.');
        }
      } else {
        alert('서버 연결 중 오류가 발생했습니다. 다시 시도해주세요.');
      }
    }
  };

  return (
    <div className="register-container">
      <h1 className="register-title">
        <span className="pink">회원정보</span>
        <span className="brown">를 입력해주세요</span>
      </h1>

      <div className="input-group">
        <input 
          type="text"
          name="username"
          placeholder="ID"
          className={`input-box ${errors.username ? 'error' : ''}`}
          value={formData.username}
          onChange={handleChange}
        />
        {errors.username && <p className="error-message">{errors.username}</p>}
      </div>

      <div className="input-group">
        <input 
          type="password"
          name="password"
          placeholder="비밀번호"
          className={`input-box ${errors.password ? 'error' : ''}`}
          value={formData.password}
          onChange={handleChange}
        />
        {errors.password && <p className="error-message">{errors.password}</p>}
      </div>

      <div className="input-group">
        <input 
          type="password"
          name="confirmPassword"
          placeholder="비밀번호 재확인"
          className={`input-box ${errors.confirmPassword ? 'error' : ''}`}
          value={formData.confirmPassword}
          onChange={handleChange}
        />
        {errors.confirmPassword && <p className="error-message">{errors.confirmPassword}</p>}
      </div>

      <div className="input-group">
        <input 
          type="text"
          name="name"
          placeholder="이름"
          className={`input-box ${errors.name ? 'error' : ''}`}
          value={formData.name}
          onChange={handleChange}
        />
        {errors.name && <p className="error-message">{errors.name}</p>}
      </div>

      <div className="input-group">
        <input 
          type="text"
          name="phone"
          placeholder="휴대폰 번호 (예: 01012345678)"
          className={`input-box ${errors.phone ? 'error' : ''}`}
          value={formData.phone}
          onChange={handleChange}
        />
        {errors.phone && <p className="error-message">{errors.phone}</p>}
      </div>

      <div className="input-group">
        <input 
          type="email"
          name="email"
          placeholder="이메일"
          className={`input-box ${errors.email ? 'error' : ''}`}
          value={formData.email}
          onChange={handleChange}
        />
        {errors.email && <p className="error-message">{errors.email}</p>}
      </div>

      <div className="button-group">
        <button className="register-back-button" onClick={() => navigate('/')}>뒤로가기</button>
        <button className="next-button" onClick={handleSubmit}>다음</button>
      </div>
    </div>
  );
}

export default RegisterPage;
