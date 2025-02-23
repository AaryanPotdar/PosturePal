# 🧍‍♂️ PosturePal  

## 🌟 Inspiration  
Good posture is essential for long-term health, yet many people struggle to maintain it, especially while working at a desk for extended periods. We wanted to build a solution that provides real-time feedback and actionable insights to help users develop better posture habits.  

---

## 🎯 What It Does  
PosturePal analyzes a user's posture using their webcam and provides instant feedback. It detects key body landmarks, calculates posture metrics, and assigns a score based on alignment.  

🚀 **Features:**  
- Detects and evaluates posture in real-time using machine learning.  
- Generates summaries and personalized improvement strategies using **AWS Bedrock** and **Mistral AI (Mistral Large 24.02)**.  
- Tracks posture history with **visualizations** for long-term insights.  

---

## 🛠️ How We Built It  
We developed PosturePal using a **full-stack approach**:  

### 📌 Frontend  
- Built with **React, TypeScript, and Tailwind CSS** for a responsive and intuitive UI.  

### 🔧 Backend  
- Flask API for authentication, posture data processing, and AWS integration.  

### 🧠 Machine Learning  
- **TensorFlow.js** and **MediaPipe** power real-time posture analysis directly in the browser.  
- Trained a custom **Neural Network** for enhanced accuracy.  

### ☁️ AWS Bedrock Integration  
- Runs **Mistral Large** to summarize posture data and generate personalized improvement strategies.  

### 🗄️ Database  
- **Supabase (PostgreSQL)** stores user data, posture scores, and session history.  

---

## 🚧 Challenges We Ran Into  
🔹 Fine-tuning the posture detection model to minimize inaccuracies.  
🔹 Ensuring **real-time performance** while integrating TensorFlow.js, MediaPipe, and AWS Bedrock.  
🔹 Optimizing **Mistral AI**'s recommendations by testing different prompt structures.  

---

## 🏆 Accomplishments That We're Proud Of  
✅ **Real-time posture tracking** with machine learning.  
✅ **AWS Bedrock + Mistral AI** for personalized posture improvement strategies.  
✅ Seamless **user experience** with authentication, tracking, and analytics.  
✅ **Google OAuth** integration for secure login.  

---

## 📚 What We Learned  
🔹 Real-time **computer vision** and AI-powered summarization.  
🔹 **Optimizing ML models** for web applications.  
🔹 Hands-on experience with **AWS Bedrock** and deploying scalable AI solutions.  

---

## 🚀 What's Next for PosturePal  
🚀 **Refining** the posture scoring system with better data and feedback.  
📱 **Expanding device compatibility**, including mobile support.  
💡 **Enhancing Mistral’s recommendations** for more personalized guidance.  

**PosturePal is more than just an app—it’s a lifestyle change powered by AI.** 🚀  

---

## 🎯 Tracks  
**Health**


