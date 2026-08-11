import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Page Setup
st.set_page_config(page_title="Smart Agriculture System", page_icon="🌱", layout="wide")

# Header Section
st.title("🌱 Smart Agriculture & Crop Recommendation System")
st.write("An AI/ML based web application for soil analysis and crop prediction.")
st.write("---")

# Student Details
st.sidebar.title("📌 Student Details")
st.sidebar.write("**Name:** Sneha Kumari")
st.sidebar.write("**Course:** AI & ML")
st.sidebar.write("**Project:** Smart Agriculture Systems")
st.sidebar.write("---")

# ML Model Training Function
@st.cache_resource
def build_ml_model():
    dataset = [
        [90, 42, 43, 20.8, 82.0, 6.5, 202.9, 'Rice'],
        [80, 40, 40, 22.0, 80.0, 6.0, 190.0, 'Rice'],
        [20, 60, 20, 21.0, 65.0, 5.5, 100.0, 'Maize'],
        [25, 55, 25, 23.5, 60.0, 6.2, 95.0, 'Maize'],
        [40, 68, 80, 18.0, 20.0, 5.8, 80.0, 'Chickpea'],
        [120, 40, 20, 25.0, 70.0, 6.8, 150.0, 'Cotton'],
        [100, 15, 30, 24.0, 85.0, 6.7, 220.0, 'Coffee']
    ]
    df = pd.DataFrame(dataset, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop'])
    X = df.drop('crop', axis=1)
    y = df['crop']
    rf_model = RandomForestClassifier(n_estimators=15, random_state=42)
    rf_model.fit(X, y)
    return rf_model

model = build_ml_model()

# Sliders Section
st.sidebar.header("⚙️ Soil & Climate Parameters")
n_val = st.sidebar.slider("Nitrogen (N)", 0, 140, 85)
p_val = st.sidebar.slider("Phosphorus (P)", 0, 145, 45)
k_val = st.sidebar.slider("Potassium (K)", 0, 205, 40)
temp = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 23.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, 75.0)
ph = st.sidebar.slider("Soil pH Level", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 180.0)

# Display Dashboard
st.subheader("📊 Soil Parameter Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("N-P-K Ratio", f"{n_val} - {p_val} - {k_val}")
col2.metric("Temperature / Humidity", f"{temp}°C | {humidity}%")
col3.metric("pH & Rainfall", f"{ph} pH | {rainfall} mm")

st.write("")

# Prediction Trigger
if st.button("🚀 Analyze Soil & Recommend Crop"):
    user_inputs = [[n_val, p_val, k_val, temp, humidity, ph, rainfall]]
    result = model.predict(user_inputs)[0]
    st.success(f"🌱 **Recommended Crop for Cultivation:** {result}")
    st.balloons()
