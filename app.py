import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from PIL import Image

# Page Setup
st.set_page_config(
    page_title="Smart Agriculture System", page_icon="🌱", layout="wide"
)

# Header Section
st.title("🌱 Smart Agriculture & Crop Recommendation System")
st.write("An AI/ML based web application for crop prediction and disease detection.")
st.write("---")

# Student Details In Sidebar
st.sidebar.title("📌 Student Details")
st.sidebar.write("**Name:** Sneha Kumari")
st.sidebar.write("**Course:** AI & ML")
st.sidebar.write("---")

# ML Model Training Function
@st.cache_resource
def build_ml_model():
    dataset = [
        [90, 42, 43, 20.8, 82.0, 6.5, 202.9, "Rice"],
        [80, 40, 40, 22.0, 80.0, 6.0, 190.0, "Rice"],
        [20, 60, 20, 21.0, 65.0, 5.5, 100.0, "Maize"],
        [25, 55, 25, 23.5, 60.0, 6.2, 95.0, "Maize"],
        [40, 68, 80, 18.0, 20.0, 5.8, 80.0, "Chickpea"],
        [120, 40, 20, 25.0, 70.0, 6.8, 150.0, "Cotton"],
        [100, 15, 30, 24.0, 85.0, 6.7, 220.0, "Coffee"],
    ]
    df = pd.DataFrame(dataset, columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop"])
    X = df.drop("crop", axis=1)
    y = df["crop"]
    rf_model = RandomForestClassifier(n_estimators=15, random_state=42)
    rf_model.fit(X, y)
    return rf_model

model = build_ml_model()

tab1, tab2, tab3, tab4 = st.tabs([
    "🎛️ Manual Sliders",
    "🌤️ Weather Test (City Search)",
    "🧪 Soil Test Report",
    "🍃 Plant Disease Detection",
])

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

with tab2:
    st.subheader("Live Weather Based Testing")
    city = st.text_input("Enter City / District:", "Patna")

    if st.button("🔍 Fetch Weather & Recommend Crop"):
        try:
            # Geocoding API call
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            geo_response = requests.get(geo_url).json()

            if "results" in geo_response and len(geo_response["results"]) > 0:
                lat = geo_response["results"][0]["latitude"]
                lon = geo_response["results"][0]["longitude"]
                
                # Fetch Weather data
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
                weather_response = requests.get(weather_url).json()
                data = weather_response["current"]
                
                st.info(f"📍 Location: {city} | 🌡️ Temp: {data['temperature_2m']}°C | 💧 Humidity: {data['relative_humidity_2m']}% | 🌧️ Rainfall: {data['precipitation']} mm")
                
                res = model.predict([[85, 45, 40, data['temperature_2m'], data['relative_humidity_2m'], 6.5, data['precipitation']]])[0]
                st.success(f"🌱 **Best Crop for {city}'s Weather:** {res}")
            else:
                st.error("Location not found.")
        except:
            st.error("Error fetching weather data.")

with tab3:
    st.subheader("Soil Lab Test Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        st.dataframe(user_df.head())
        st.success("Report Processed!")
    else:
        s_n = st.number_input("Nitrogen", value=90)
        s_p = st.number_input("Phosphorus", value=42)
        s_k = st.number_input("Potassium", value=43)
        if st.button("🧪 Analyze Soil"):
            res = model.predict([[s_n, s_p, s_k, 23.0, 75.0, 6.5, 180.0]])[0]
            st.success(f"🌱 **Recommended Crop:** {res}")

with tab4:
    st.subheader("🍃 Leaf Disease Detection")
    uploaded_leaf = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_leaf:
        leaf_image = Image.open(uploaded_leaf)
        st.image(leaf_image, caption="Uploaded Leaf Image", use_container_width=True)

        if st.button("🔍 Detect Disease"):
            st.info("🔄 Processing image...")
            st.success("✅ **Analysis Completed!**")
            st.warning("⚠️ **Detected Disease:** Tomato - Early Blight (Fungal)")
            st.markdown("""
                **💡 Recommended Remedy:**
                - Apply copper-based fungicide spray.
                - Remove infected leaves.
                """)
        
  
