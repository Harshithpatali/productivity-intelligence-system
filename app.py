import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from src.model_training import NeuralNetwork
from src.config import *

# =============================
# Load Artifacts
# =============================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_columns = joblib.load("models/feature_columns.pkl")

# =============================
# Page Config
# =============================

st.set_page_config(
    page_title="HR Productivity AI",
    layout="wide",
    page_icon="🏢"
)

# =============================
# Custom HR CSS
# =============================

st.markdown("""
<style>

.main {
    background: linear-gradient(120deg, #eef2f7, #f8fbff);
}

.card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

.title {
    font-size: 34px;
    font-weight: 700;
    color: #1f3b73;
}

.subtitle {
    font-size: 18px;
    color: #5a6a85;
}

.metric-box {
    background: #f4f7fc;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =============================
# Header
# =============================

st.markdown("""
<div class="card">
<div class="title">🏢 Employee Productivity Intelligence Platform</div>
<div class="subtitle">AI-Driven Cognitive Load & Performance Monitoring</div>
</div>
""", unsafe_allow_html=True)

# =============================
# Sidebar Inputs
# =============================

st.sidebar.header("Employee Behavior Inputs")

sleep = st.sidebar.slider("Sleep Hours", 3.0, 10.0, 7.0)
work = st.sidebar.slider("Work Hours", 2.0, 14.0, 8.0)
screen = st.sidebar.slider("Screen Time", 1.0, 12.0, 6.0)
breaks = st.sidebar.slider("Breaks Taken", 0, 10, 4)
switches = st.sidebar.slider("Task Switches", 0, 30, 10)
stress = st.sidebar.slider("Stress Level", 1.0, 10.0, 5.0)
hydration = st.sidebar.slider("Hydration Level", 1.0, 5.0, 3.0)
noise = st.sidebar.slider("Noise Level", 1.0, 10.0, 5.0)

# =============================
# Create Input Data
# =============================

data = pd.DataFrame([{
    "sleep_hours": sleep,
    "work_hours": work,
    "screen_time": screen,
    "breaks": breaks,
    "task_switches": switches,
    "stress_level": stress,
    "hydration": hydration,
    "noise_level": noise
}])

# =============================
# Feature Engineering (same as training)
# =============================

data["work_sleep_ratio"] = data["work_hours"] / (data["sleep_hours"] + 1)
data["stress_per_hour"] = data["stress_level"] / (data["work_hours"] + 1)
data["break_efficiency"] = data["breaks"] / (data["work_hours"] + 1)
data["focus_index"] = data["hydration"] / (data["task_switches"] + 1)
data["screen_fatigue"] = data["screen_time"] * data["stress_level"]

# Add placeholder columns that existed during training
data["productivity_score"] = 0
data["fatigue_score"] = 0

# =============================
# Align Feature Order
# =============================

for col in feature_columns:
    if col not in data.columns:
        data[col] = 0

data = data[feature_columns]

# =============================
# Scale & Predict
# =============================

X_scaled = scaler.transform(data)

probs = model.forward(X_scaled, training=False)
pred_class = np.argmax(probs, axis=1)
label = encoder.inverse_transform(pred_class)[0]

# =============================
# Fatigue Score
# =============================

fatigue_score = (
    work * 6 +
    stress * 8 -
    sleep * 7 -
    breaks * 3
)

fatigue_score = max(0, min(100, fatigue_score))

# =============================
# Metrics Layout
# =============================

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.subheader("📊 Productivity Level")
    st.metric("Prediction", label)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.subheader("⚠️ Fatigue Risk Score")
    st.metric("Fatigue", f"{fatigue_score:.2f}")
    st.progress(fatigue_score / 100)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# Probability Chart
# =============================

st.markdown("### 📈 Productivity Probability Distribution")

prob_df = pd.DataFrame(probs, columns=encoder.classes_)
st.bar_chart(prob_df.T)

# =============================
# Behavioral Chart
# =============================

st.markdown("### 📊 Employee Behavior Overview")

fig, ax = plt.subplots()

features = [
    sleep,
    work,
    screen,
    breaks,
    switches,
    stress,
    hydration,
    noise
]

labels = [
    "Sleep",
    "Work",
    "Screen",
    "Breaks",
    "Switches",
    "Stress",
    "Hydration",
    "Noise"
]

ax.bar(labels, features)
plt.xticks(rotation=45)

st.pyplot(fig)

# =============================
# Footer
# =============================

st.markdown("---")
st.markdown("HR Analytics AI System • Built with Custom Neural Network")
