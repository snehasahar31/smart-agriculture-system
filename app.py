import base64, hashlib, os
import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# --- 1. Page Configuration ---
st.set_page_config(page_title="Smart Agriculture System", page_icon="🌱", layout="wide")

# --- 2. User Authentication & Database Setup ---
make_hash = lambda p: hashlib.sha256(p.encode()).hexdigest()
st.session_state.setdefault("user_db", {"admin": make_hash("admin123"), "farmer": make_hash("farmer123")})
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("username", "")

# --- 3. Login & Registration Screen ---
if not st.session_state["logged_in"]:
    st.title("🌱 Smart Agriculture Portal")
    t_login, t_reg = st.tabs(["🔐 Login", "📝 Register"])

    # Login Section
    with t_login:
        u = st.text_input("Username", key="l_u")
        p = st.text_input("Password", type="password", key="l_p")
        if st.button("Login", key="l_btn"):
            if u in st.session_state["user_db"] and st.session_state["user_db"][u] == make_hash(p):
                st.session_state.update({"logged_in": True, "username": u})
                st.rerun()
            else:
                st.error("Invalid Credentials!")

    # Registration Section
    with t_reg:
        ru = st.text_input("Choose Username", key="r_u")
        rp = st.text_input("Choose Password", type="password", key="r_p")
        rcp = st.text_input("Confirm Password", type="password", key="r_cp")
        if st.button("Register Account", key="r_btn"):
            if not ru.strip() or not rp.strip():
                st.warning("Fill all fields.")
            elif ru in st.session_state["user_db"]:
                st.error("Username taken.")
            elif rp != rcp:
                st.error("Passwords do not match!")
            else:
                st.session_state["user_db"][ru] = make_hash(rp)
                st.success("Account created successfully!")

# --- 4. Main Application Dashboard ---
else:
    # Sidebar User Profile & Logout
    st.sidebar.markdown(f"👤 **Logged User:** `{st.session_state['username']}`")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.update({"logged_in": False, "username": ""})
        st.rerun()

    st.title("🌱 Smart Agriculture & Crop Recommendation System")
    
    # --- ML Crop Model Training ---
    FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    @st.cache_resource
    def get_trained_crop_model():
        file_path = "Crop_recommendation.csv"
        if not os.path.exists(file_path):
            crops = {
                "Rice": (80,100, 35,60, 35,45, 20,27, 80,90, 5.5,7.2, 180,300),
                "Maize": (60,100, 35,60, 15,25, 18,27, 55,70, 5.5,7.0, 60,110),
                "Chickpea": (20,50, 55,80, 70,85, 17,20, 14,20, 5.5,6.5, 65,95),
                "Cotton": (100,140, 35,60, 15,25, 22,26, 60,80, 5.8,8.0, 60,110),
                "Coffee": (80,120, 15,35, 25,35, 23,27, 50,80, 6.0,7.0, 115,200),
                "Wheat": (40,70, 40,60, 40,60, 15,25, 50,70, 6.0,7.5, 70,130),
                "Sugarcane": (90,120, 30,50, 25,40, 26,32, 55,70, 6.0,7.5, 80,110),
                "Banana": (90,120, 70,95, 45,55, 25,30, 75,85, 5.5,6.5, 90,120),
                "Apple": (10,40, 120,145, 195,205, 21,24, 90,95, 5.5,6.5, 100,125),
                "Papaya": (35,60, 45,70, 45,55, 23,35, 80,95, 6.5,7.0, 80,200)
            }
            rows = []
            for crop, b in crops.items():
                for _ in range(30):
                    rows.append([
                        np.random.randint(b[0], b[1]), np.random.randint(b[2], b[3]), np.random.randint(b[4], b[5]),
                        round(np.random.uniform(b[6], b[7]), 1), round(np.random.uniform(b[8], b[9]), 1),
                        round(np.random.uniform(b[10], b[11]), 1), round(np.random.uniform(b[12], b[13]), 1), crop
                    ])
            pd.DataFrame(rows, columns=FEATURE_COLS + ["crop"]).to_csv(file_path, index=False)

        df = pd.read_csv(file_path)
        return RandomForestClassifier(n_estimators=100, random_state=42).fit(df[FEATURE_COLS], df["crop"])

    crop_model = get_trained_crop_model()
    predict_crop = lambda n, p, k, t, h, ph, r: crop_model.predict(pd.DataFrame([[n, p, k, t, h, ph, r]], columns=FEATURE_COLS))[0]

    # --- Plant.id Disease Detection API Helper ---
    def check_plant_disease_api(image_bytes, api_key):
        img_b64 = base64.b64encode(image_bytes).decode('utf-8')
        url = "https://plant.id/api/v3/health_assessment"
        payload = {"images": [f"data:image/jpeg;base64,{img_b64}"], "similar_images": True}
        return requests.post(url, json=payload, headers={"Api-Key": api_key, "Content-Type": "application/json"}, timeout=15)

    # --- Live Weather API Helper ---
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_weather(city: str):
        try:
            g = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}, timeout=5).json()
            loc = g["results"][0]
            w = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": loc["latitude"], "longitude": loc["longitude"], "current": "temperature_2m,relative_humidity_2m,precipitation"}, timeout=5).json()
            return {"name": loc.get("name", city), "temp": w["current"]["temperature_2m"], "hum": w["current"]["relative_humidity_2m"], "rain": w["current"]["precipitation"]}
        except Exception:
            return None

    # --- Main Navigation Tabs ---
    t1, t2, t3, t4 = st.tabs(["🎛️ Manual Sliders", "🌤️ Weather Search", "📊 Soil Test Report", "🍃 Disease Detection (Plant.id API)"])

    # --- Tab 1: Manual Parameter Testing (Sliders) ---
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            n, p, k = st.slider("Nitrogen (N)", 0, 140, 85), st.slider("Phosphorus (P)", 0, 145, 45), st.slider("Potassium (K)", 0, 205, 40)
            ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5)
        with c2:
            temp, hum, rain = st.slider("Temp (°C)", 10.0, 50.0, 23.0), st.slider("Humidity (%)", 10.0, 100.0, 75.0), st.slider("Rainfall (mm)", 20.0, 300.0, 180.0)
        if st.button("Predict Crop", key="b1"):
            st.success(f"**Recommended Crop:** {predict_crop(n, p, k, temp, hum, ph, rain)}")

    # --- Tab 2: Live Weather Based Recommendation ---
    with t2:
        city = st.text_input("Enter City / District:", "Patna")
        c1, c2 = st.columns(2)
        with c1:
            avg_rain = st.slider("Average Seasonal Rainfall (mm):", 30, 300, 100)
        with c2:
            soil_type = st.selectbox("Soil Profile Preset:", ["Standard Soil (Balanced)", "High Nitrogen Soil", "P/K Rich Soil"])

        if st.button("Fetch Weather & Predict", key="b2"):
            w = fetch_weather(city.strip())
            if w:
                st.info(f"📍 Location: {w['name']} | Temp: {w['temp']}°C | Humidity: {w['hum']}%")
                
                # Preset soil values based on selection
                if soil_type == "High Nitrogen Soil":
                    sn, sp, sk = 110, 45, 30
                elif soil_type == "P/K Rich Soil":
                    sn, sp, sk = 30, 80, 80
                else:
                    sn, sp, sk = 65, 45, 45

                res_crop = predict_crop(sn, sp, sk, w['temp'], w['hum'], 6.5, avg_rain)
                st.success(f"**Recommended Crop for {w['name']}:** {res_crop}")
            else:
                st.error("Could not fetch weather data.")

    # --- Tab 3: Soil Test CSV Upload & Analysis ---
    with t3:
        up_file = st.file_uploader("Upload Soil CSV", type=["csv"])
        if up_file:
            try:
                df = pd.read_csv(up_file)
                df["Recommended_Crop"] = [predict_crop(r.get("N", 90), r.get("P", 42), r.get("K", 43), r.get("temperature", 23.0), r.get("humidity", 75.0), r.get("ph", 6.5), r.get("rainfall", 180.0)) for _, r in df.iterrows()]
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="crop_analysis.csv")
            except Exception:
                st.error("Error reading CSV file.")
        else:
            sn, sp, sk = st.number_input("N", value=90), st.number_input("P", value=42), st.number_input("K", value=43)
            if st.button("Evaluate Sample", key="b3"):
                st.success(f"**Recommended Crop:** {predict_crop(sn, sp, sk, 23.0, 75.0, 6.5, 180.0)}")

    # --- Tab 4: Leaf Disease Detection (Plant.id API) ---
    with t4:
        st.subheader("🍃 Plant Health Diagnostic (Plant.id API)")
        
        # Embedded User API Key
        PLANT_ID_API_KEY = "EI9DKae6Sgbd28Rqx2AnaIV6XmOZZxIZZo01mYljV9tXUSRLdZ"
        api_key = st.text_input("Plant.id API Key:", value=PLANT_ID_API_KEY, type="password")

        img_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
        if img_file:
            st.image(Image.open(img_file), caption="Uploaded Specimen", width=300)
            if st.button("Run Diagnostic", key="b4"):
                if api_key.strip():
                    with st.spinner("Analyzing with Plant.id API..."):
                        res = check_plant_disease_api(img_file.getvalue(), api_key.strip())
                        if res.status_code in (200, 201):
                            data = res.json().get("result", {})
                            if data.get("is_healthy", {}).get("binary", True):
                                st.info("🟢 **Plant Status:** Healthy / Normal Leaf")
                            else:
                                diseases = data.get("disease", {}).get("suggestions", [])
                                if diseases:
                                    d = diseases[0]
                                    st.error(f"🔴 **Detected Disease:** `{d.get('name')}` ({d.get('probability',0)*100:.1f}% Confidence)")
                                    for k_t, v_t in d.get("details", {}).get("treatment", {}).items():
                                        st.write(f"**{k_t.title()}:** {v_t}")
                                else:
                                    st.warning("Disease detected but unclassified.")
                        else:
                            st.error(f"API Request Failed (Status Code: {res.status_code}). Please verify your Plant.id API key/credits.")
                else:
                    st.warning("⚠️ Enter a valid Plant.id API Key above.")
        
