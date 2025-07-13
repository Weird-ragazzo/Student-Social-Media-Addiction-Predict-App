# 🎓 Student Social Media Addiction Predictor (Multi-Model App)

An interactive and modern **Streamlit web app** that predicts whether a student is addicted to social media using one of four powerful machine learning models. Built and designed by **Dhruv Raghav**, this app integrates multiple models into a single user-friendly dashboard.

---

## 🔍 What It Does

- Takes input on student behavior, habits, and social media usage
- Allows the user to select one of the following ML models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Shows real-time predictions with styled results
- Displays the accuracy of each model
- Easy-to-use, responsive interface

---

## 🧠 Models & Accuracy

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 99.0%    |
| Random Forest       | 100.0%   |
| XGBoost             | 99.2%    |
| LightGBM            | 99.4%    |

All models are pre-trained and stored in the `models/` directory as `.pkl` files.

---

## 📊 Features

- Modern, interactive UI using Streamlit  
- Individual prediction apps for each model  
- User input form for 12+ features  
- Result display with visual feedback  
- Model accuracy info embedded in each app

---

## 🧠 Dataset & Features

The model was trained on a dataset with the following features:

- Gender  
- Academic Level  
- Average Daily Usage (hours)  
- Affects Academic Performance  
- Sleep Hours per Night  
- Mental Health Score (1–10)  
- Relationship Status  
- Conflicts Over Social Media  
- Most Used Social Media Platform (12 options)

**Target variable:** `Addicted Score (binary)`

---

## 👨‍💻 Author

**Dhruv Raghav**


