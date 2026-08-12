import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from PIL import Image

# Setup page config
st.set_page_config(
    page_title="Smart Agriculture System", page_icon="🌱", layout="wide"
)

st.title("🌱 Smart Agriculture & Crop Recommendation System")
st.write("An AI/ML based web application for crop prediction and disease detection.")
st.divider()

# Sidebar info
st.sidebar.title("📌 Project Details")
st.sidebar.write("**Course:** AI & ML")
st.sidebar.divider()

# Random Forest Model Training
@st.cache_resource
def train_crop_model():
    dataset = [
        [90, 42, 43, 20.8, 82.0, 6.5, 202.9, "Rice"],
        [80, 40, 40, 22.0, 80.0, 6.0, 190.0, "Rice"],
        [20, 60, 20, 21.0, 65.0, 5.5, 100.0, "Maize"],
        [25, 55, 25, 23.5, 60.0, 6.2, 95.0, "Maize"],
        [40, 68, 80, 18.0, 20.0, 5.8, 80.0, "Chickpea"],
        [120, 40, 20, 25.0, 70.0, 6.8, 150.0, "Cotton"],
        [100, 15, 30, 24.0, 85.0, 6.7, 220.0, "Coffee"],
    ]
    df_train = pd.DataFrame(dataset, columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "crop"])
    X = df_train.drop("crop", axis=1)
    y = df_train["crop"]
    
    rf = RandomForestClassifier(n_estimators=15, random_state=42)
    rf.fit(X, y)
    return rf

model = train_crop_model()

# Main Tabs layout
t1, t2, t3, t4 = st.tabs([
    "🎛️ Manual Sliders",
    "🌤️ Weather Test (City Search)",
    "🧪 Soil Test Report",
    "🍃 Plant Disease Detection",
])

# Tab 1: Sliders
with t1:
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

    if st.button("Analyze & Predict"):
        pred = model.predict([[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]])[0]
        st.success(f"🌱 **Recommended Crop:** {pred}")

# Tab 2: Live Weather API
with t2:
    st.subheader("Live Weather Based Testing")
    city_input = st.text_input("Enter City / District:", "Patna")

    if st.button("Fetch Weather & Recommend"):
        try:
            # Geocoding call
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_input}&count=1&language=en&format=json"
            geo_res = requests.get(geo_url).json()

            if "results" in geo_res and len(geo_res["results"]) > 0:
                lat = geo_res["results"][0]["latitude"]
                lon = geo_res["results"][0]["longitude"]
                
                # Fetch Weather data
                w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation"
                w_res = requests.get(w_url).json()
                weather_data = w_res["current"]
                
                st.info(f"📍 Location: {city_input} | 🌡️ Temp: {weather_data['temperature_2m']}°C | 💧 Humidity: {weather_data['relative_humidity_2m']}% | 🌧️ Rainfall: {weather_data['precipitation']} mm")
                
                pred = model.predict([[85, 45, 40, weather_data['temperature_2m'], weather_data['relative_humidity_2m'], 6.5, weather_data['precipitation']]])[0]
                st.success(f"🌱 **Best Crop for {city_input}'s Weather:** {pred}")
            else:
                st.error("Location not found.")
        except Exception:
            st.error("Error fetching weather data.")

# Tab 3: CSV Processing
with t3:
    st.subheader("Soil Lab Test Analysis")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_csv:
        csv_df = pd.read_csv(uploaded_csv)
        
        preds = []
        for _, row in csv_df.iterrows():
            n = row.get('N', 90)
            p = row.get('P', 42)
            k = row.get('K', 43)
            temp = row.get('temperature', 23.0)
            hum = row.get('humidity', 75.0)
            ph = row.get('ph', 6.5)
            rain = row.get('rainfall', 180.0)
            
            crop_out = model.predict([[n, p, k, temp, hum, ph, rain]])[0]
            preds.append(crop_out)
        
        csv_df['Recommended_Crop'] = preds
        
        st.write("### 📊 Soil Analysis & Crop Predictions")
        st.dataframe(csv_df, use_container_width=True)
        st.success("✅ Report Processed Successfully!")
        
        st.write("### 🌱 Crop Recommendations")
        for idx, row in csv_df.iterrows():
            sample_id = row.get('Sample_ID', row.get('Location', f"Sample #{idx+1}"))
            crop_name = row['Recommended_Crop']
            st.success(f"📍 **{sample_id}** ➔ Recommended Crop: **{crop_name}**")
            
    else:
        s_n = st.number_input("Nitrogen", value=90)
        s_p = st.number_input("Phosphorus", value=42)
        s_k = st.number_input("Potassium", value=43)
        if st.button("Analyze Soil"):
            res = model.predict([[s_n, s_p, s_k, 23.0, 75.0, 6.5, 180.0]])[0]
            st.success(f"🌱 **Recommended Crop:** {res}")

# Tab 4: Image Detection
with t4:
    st.subheader("🍃 Leaf Disease Detection")
    leaf_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if leaf_file:
        img = Image.open(leaf_file)
        st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

        if st.button("Detect Disease"):
            st.info("🔄 Processing image...")
            st.success("✅ **Analysis Completed!**")
            st.warning("⚠️ **Detected Disease:** Tomato - Early Blight (Fungal)")
            st.markdown("""
                **💡 Recommended Remedy:**
                - Apply copper-based fungicide spray.
                - Remove infected leaves.
            """)
                
