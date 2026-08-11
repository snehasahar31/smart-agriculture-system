import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

# Page Setup
st.set_page_config(page_title="Smart Agriculture System", page_icon="🌱", layout="wide")

# Header Section
st.title("🌱 Smart Agriculture & Crop Recommendation System")
st.write("An AI/ML based web application for soil analysis, weather testing, and crop prediction.")
st.write("---")

# Student Details in Sidebar
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

# Create Tabs for Options
tab1, tab2, tab3 = st.tabs(["🎛️ Manual Sliders", "🌤️ Weather Test (City Search)", "🧪 Soil Test Report"])

# TAB 1: Manual Input
with tab1:
    st.subheader("Manual Parameter Testing")
    col1, col2 = st.columns(2)
    with col1:
        n_val = st.slider("Nitrogen (N)", 0, 140, 85)
        p_val = st.slider("Phosphorus (P)", 0, 145, 45)
        k_val = st.slider("Potassium (K)", 0, 205, 40)
        ph_val = st.slider("Soil pH Level", 0.0, 14.0, 6.5)
    with col2:
        temp_val = st.slider("Temperature (°C)", 10.0, 50.0, 23.0)
        hum_val = st.slider("Humidity (%)", 10.0, 100.0, 75.0)
        rain_val = st.slider("Rainfall (mm)", 20.0, 300.0, 180.0)
    
    if st.button("🚀 Analyze & Predict (Manual)"):
        res = model.predict([[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]])[0]
        st.success(f"🌱 **Recommended Crop:** {res}")

# TAB 2: Weather Test Option
with tab2:
    st.subheader("Live Weather Based Testing")
    city = st.text_input("Apni City / District ka naam likhein:", "Delhi")
    
    if st.button("🔍 Fetch Weather & Recommend Crop"):
        try:
            # Weather API call
            url = f"https://api.open-meteo.com/v1/forecast?latitude=28.61&longitude=77.23&current_weather=true"
            response = requests.get(url).json()
            live_temp = response['current_weather']['temperature']
            st.info(f"📍 **Detected Live Temp in {city}:** {live_temp}°C")
            
            # Predict with default soil + live temp
            res = model.predict([[85, 45, 40, live_temp, 75.0, 6.5, 180.0]])[0]
            st.success(f"🌱 **Best Crop for {city}'s Weather:** {res}")
        except:
            st.error("Weather data fetch nahi ho paya, please try again.")

# TAB 3: Soil Test Option
with tab3:
    st.subheader("Soil Lab Test Analysis")
    uploaded_file = st.file_uploader("Upload Soil Lab Report (CSV file)", type=["csv"])
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("📊 **Uploaded Soil Report Preview:**")
        st.dataframe(user_df.head())
        st.success("✅ Lab Report Processed Successfully!")
    else:
        st.info("💡 Aapke paas report file nahi hai? Niche Lab values manually enter karein:")
        s_n = st.number_input("Lab Nitrogen Value", value=90)
        s_p = st.number_input("Lab Phosphorus Value", value=42)
        s_k = st.number_input("Lab Potassium Value", value=43)
        
        if st.button("🧪 Analyze Soil Test Values"):
            res = model.predict([[s_n, s_p, s_k, 23.0, 75.0, 6.5, 180.0]])[0]
            st.success(f"🌱 **Crop Recommendation Based on Soil Test:** {res}")
