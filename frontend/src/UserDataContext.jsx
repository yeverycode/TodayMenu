import React, { createContext, useContext, useState } from 'react';

const UserDataContext = createContext();

export function UserDataProvider({ children }) {
  const [allergy, setAllergy] = useState([]);
  const [disease, setDisease] = useState([]);
  const [preferredMenu, setPreferredMenu] = useState([]);
  const [dislikedMenu, setDislikedMenu] = useState([]);

  const value = {
    allergy, setAllergy,
    disease, setDisease,
    preferredMenu, setPreferredMenu,
    dislikedMenu, setDislikedMenu
  };

  return (
    <UserDataContext.Provider value={value}>
      {children}
    </UserDataContext.Provider>
  );
}

export function useUserData() {
  return useContext(UserDataContext);
}
