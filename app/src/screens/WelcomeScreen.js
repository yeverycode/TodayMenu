import React from 'react';
import { View, Text, Image, TouchableOpacity, StyleSheet } from 'react-native';

export default function WelcomeScreen() {
  return (
    <View style={styles.container}>
      <Image source={require('./assets/chef.png')} style={styles.character} />
      <Text style={styles.title}>
        오늘의 <Text style={styles.highlight}>먹방은</Text>
      </Text>
      <TouchableOpacity style={styles.startButton}>
        <Text style={styles.startButtonText}>시작하기</Text>
      </TouchableOpacity>
      <Text style={styles.loginText}>
        이미 계정이 있나요? <Text style={styles.loginLink}>로그인</Text>
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fffbe9',
    alignItems: 'center',
    justifyContent: 'center',
  },
  character: {
    width: 200,
    height: 200,
    resizeMode: 'contain',
  },
  title: {
    fontSize: 40,
    color: '#e89aa8',
    marginVertical: 20,
  },
  highlight: {
    color: '#b57b47',
  },
  startButton: {
    backgroundColor: '#f2b6bd',
    borderRadius: 30,
    paddingVertical: 15,
    paddingHorizontal: 40,
    marginVertical: 30,
  },
  startButtonText: {
    color: 'white',
    fontSize: 20,
  },
  loginText: {
    fontSize: 16,
    color: '#b57b47',
  },
  loginLink: {
    color: '#e89aa8',
  },
});
