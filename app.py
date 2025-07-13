import streamlit as st
import numpy as np
import joblib

# Set page config
st.set_page_config(
    page_title="Student Addiction Predictor", page_icon="📊", layout="centered"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f8fa;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #0072C6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stSelectbox label {
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class='main'>", unsafe_allow_html=True)

# App Title
st.title("📘 Student Social Media Addiction Predictor")
st.markdown("A project by **Dhruv Raghav**")

# Model Selection
st.subheader("🔍 Choose a Model")

model_options = {
    "Logistic Regression (Accuracy: 99%)": ("student_addiction_model_LogReg.pkl", 99.0),
    "Random Forest (Accuracy: 100%)": ("model_RandomForest.pkl", 100.0),
    "XGBoost (Accuracy: 99.2%)": ("model_XGBoost.pkl", 99.2),
    "LightGBM (Accuracy: 99.4%)": ("model_LightGBM.pkl", 99.4),
}

selected_model_name = st.selectbox("Select a model", list(model_options.keys()))
model_path, accuracy = model_options[selected_model_name]
model = joblib.load(model_path)

st.success(f"✅ **{selected_model_name}** selected with accuracy **{accuracy}%**")

# Input Form
st.subheader("📥 Enter Student Details")

gender = st.selectbox("Gender", ["Male", "Female"])
academic_level = st.selectbox(
    "Academic Level", ["High School", "Undergraduate", "Graduate"]
)
avg_daily_usage = st.slider("Average Daily Usage (hours)", 0.0, 12.0, 2.0, step=0.1)
affects_academic = st.selectbox("Affects Academic Performance?", ["No", "Yes"])
sleep_hours = st.slider("Sleep Hours Per Night", 0.0, 12.0, 6.0, step=0.5)
mental_health = st.slider("Mental Health Score (1–10)", 1, 10, 5)
relationship_status = st.selectbox(
    "Relationship Status", ["Single", "In a Relationship", "Other"]
)
conflicts = st.slider("Conflicts Over Social Media", 0, 10, 0)

platforms = [
    "Facebook",
    "Instagram",
    "KakaoTalk",
    "LINE",
    "LinkedIn",
    "Snapchat",
    "TikTok",
    "Twitter",
    "VKontakte",
    "WeChat",
    "WhatsApp",
    "YouTube",
]
most_used_platform = st.selectbox("Most Used Platform", platforms)

# Encoding input
gender_val = 1 if gender == "Male" else 0
academic_val = {"High School": 1, "Undergraduate": 2, "Graduate": 3}[academic_level]
affects_val = 1 if affects_academic == "Yes" else 0
rel_val = {"Single": 1, "In a Relationship": 2, "Other": 3}[relationship_status]
platform_encoded = [
    1 if platform == most_used_platform else 0 for platform in platforms
]

input_features = np.array(
    [
        gender_val,
        academic_val,
        avg_daily_usage,
        affects_val,
        sleep_hours,
        mental_health,
        rel_val,
        conflicts,
    ]
    + platform_encoded
).reshape(1, -1)

# Predict button
if st.button("🚀 Predict Addiction"):
    prediction = model.predict(input_features)[0]
    if prediction == 1:
        st.error("⚠️ The student is likely **Addicted** to social media.")
    else:
        st.success("✅ The student is **Not Addicted** to social media.")

st.markdown("</div>", unsafe_allow_html=True)
