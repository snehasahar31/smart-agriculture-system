import base64,daatetime,hashlib
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# --- 2. Page Config & Session Setup ---
st.set_page_config(page_title="Smart Agriculture System", page_icon="🌱", layout="wide")

# Password Hashing & Login Database Initialization
make_hash = lambda p: hashlib.sha256(p.encode()).hexdigest()
st.session_state.setdefault("user_db", {"admin": make_hash("admin123"), "farmer": make_hash("farmer123")})
st.session_state.setdefault("logged_in", False)

# --- 3. ML Model Features & Crop Knowledge Base ---
FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CROPS = ["Rice", "Maize", "Chickpea", "Cotton", "Coffee", "Wheat", "Sugarcane", "Banana", "Apple", "Papaya"]

# Crop Cultivation Information (Season, Water, Duration, Tips)
CROP_INFO = {
    "Rice": ("Kharif", "High", "100-150 Days", "Standing water & clayey soil required."),
    "Maize": ("Kharif", "Moderate", "90-110 Days", "Needs well-drained fertile soil."),
    "Chickpea": ("Rabi", "Low", "90-120 Days", "Requires cool weather & light soil."),
    "Cotton": ("Kharif", "Moderate", "150-180 Days", "Deep black soil with good drainage."),
    "Coffee": ("Perennial", "High", "Perennial", "Humid hill slopes under shade."),
    "Wheat": ("Rabi", "Moderate", "120-150 Days", "Cool growth, warm sunny ripening."),
    "Sugarcane": ("Perennial", "High", "12-18 Months", "Loamy soil rich in organic matter."),
    "Banana": ("All Season", "High", "12-15 Months", "Rich, well-drained moist soil."),
    "Apple": ("Temperate", "Moderate", "Perennial", "Chilling hours required in winter."),
    "Papaya": ("All Season", "Moderate", "9-12 Months", "Sensitive to waterlogging.")
}

# --- 4. ML Model Training Function ---
@st.cache_resource
def get_trained_crop_model():
    if not os.path.exists("Crop_recommendation_v2.csv"):
        # Auto-generate synthetic dataset if CSV not found
        rows = [[np.random.randint(20,120), np.random.randint(15,100), np.random.randint(15,150),
                 round(np.random.uniform(12,35), 1), round(np.random.uniform(40,95), 1),
                 round(np.random.uniform(5.5,7.5), 1), round(np.random.uniform(40,250), 1), c]
                for c in CROPS for _ in range(40)]
        pd.DataFrame(rows, columns=FEATURE_COLS + ["crop"]).to_csv("Crop_recommendation_v2.csv", index=False)
    
    df = pd.read_csv("Crop_recommendation_v2.csv")
    return RandomForestClassifier(n_estimators=100, random_state=42).fit(df[FEATURE_COLS], df["crop"])

crop_model = get_trained_crop_model()
predict_crop = lambda n, p, k, t, h, ph, r: crop_model.predict(pd.DataFrame([[n, p, k, t, h, ph, r]], columns=FEATURE_COLS))[0]

# UI Helper: Show Crop Details Card
def show_crop_details(crop_name):
    if crop_name in CROP_INFO:
        s, w, d, t = CROP_INFO[crop_name]
        with st.expander(f"📋 **Cultivation Guide for {crop_name}**", expanded=True):
            st.write(f"🗓️ **Season:** {s} | 💧 **Water:** {w} | ⏳ **Duration:** {d}\n\n💡 **Tip:** {t}")

# --- 5. External API Helpers ---
# Plant.id Leaf Disease Detection API Helper
def check_plant_disease_api(img_bytes, api_key):
    try:
        response = requests.post(
            "https://plant.id/api/v3/health_assessment", 
            json={
                "images": [f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"], 
                "similar_images": True
            }, 
            headers={"Api-Key": api_key, "Content-Type": "application/json"}, 
            timeout=15
        )
        return response
    except Exception as e:
        return None

# Open-Meteo Live Weather API Helper
@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(city: str):
    try:
        g = requests.get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}, timeout=5).json()["results"][0]
        w = requests.get("https://api.open-meteo.com/v1/forecast", params={"latitude": g["latitude"], "longitude": g["longitude"], "current": "temperature_2m,relative_humidity_2m"}, timeout=5).json()["current"]
        return {"name": g.get("name", city), "temp": w["temperature_2m"], "hum": w["relative_humidity_2m"]}
    except Exception: 
        return None

# --- 6. User Authentication Screen (Login / Register) ---
if not st.session_state["logged_in"]:
    st.title("🌱 Smart Agriculture Portal")
    t_login, t_reg = st.tabs(["🔐 Login", "📝 Register"])
    
    # Login Tab
    with t_login:
        u, p = st.text_input("Username", key="l_u"), st.text_input("Password", type="password", key="l_p")
        if st.button("Login"):
            if st.session_state["user_db"].get(u) == make_hash(p):
                st.session_state.update({"logged_in": True, "username": u})
                st.rerun()
            else: st.error("Invalid Credentials!")
            
    # Registration Tab
    with t_reg:
        ru, rp, rcp = st.text_input("Choose Username", key="r_u"), st.text_input("Choose Password", type="password", key="r_p"), st.text_input("Confirm Password", type="password", key="r_cp")
        if st.button("Register Account"):
            if ru in st.session_state["user_db"]: st.error("Username taken.")
            elif rp != rcp or not ru.strip(): st.error("Password mismatch or empty fields!")
            else:
                st.session_state["user_db"][ru] = make_hash(rp)
                st.success("Account created!")

# --- 7. Main Dashboard (After Login) ---
else:
    # Sidebar Logout
    st.sidebar.markdown(f"👤 **Logged User:** `{st.session_state['username']}`")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.update({"logged_in": False, "username": ""})
        st.rerun()

    st.title("🌱 Smart Agriculture & Crop Recommendation System")
    t1, t2, t3, t4 = st.tabs(["🎛️ Sliders", "🌤️ Weather Search", "📊 Soil Test", "🍃 Disease Detection"])

    # Tab 1: Manual Input Sliders
    with t1:
        c1, c2 = st.columns(2)
        n, p, k = c1.slider("N", 0, 140, 75), c1.slider("P", 0, 145, 45), c1.slider("K", 0, 205, 40)
        ph = c1.slider("pH", 0.0, 14.0, 6.5)
        temp, hum, rain = c2.slider("Temp (°C)", 10.0, 50.0, 23.0), c2.slider("Humidity (%)", 10.0, 100.0, 75.0), c2.slider("Rainfall (mm)", 20.0, 300.0, 180.0)
        
        if st.button("Predict Crop", key="b1"):
            res = predict_crop(n, p, k, temp, hum, ph, rain)
            st.success(f"🌾 **Recommended Crop:** {res}")
            show_crop_details(res)

    # Tab 2: Live Weather & Seasonal Prediction
    with t2:
        now = datetime.datetime.now()
        season, suitable = ("Kharif (Monsoon)", ["Rice", "Cotton", "Sugarcane", "Maize", "Papaya"]) if 6 <= now.month <= 10 else \
                           ("Rabi (Winter)", ["Wheat", "Chickpea", "Apple"]) if now.month >= 11 or now.month <= 4 else \
                           ("Zaid (Summer)", ["Coffee", "Banana", "Papaya"])
        
        st.info(f"📅 **Date:** {now.strftime('%d %B %Y')} | **Season:** {season}")
        city = st.text_input("City / District:", "Ranchi")
        c1, c2 = st.columns(2)
        avg_rain, soil = c1.slider("Avg Rainfall (mm):", 30, 300, 150), c2.selectbox("Soil Preset:", ["Balanced", "High Nitrogen", "P/K Rich"])

        if st.button("Fetch & Predict", key="b2"):
            w = fetch_weather(city.strip())
            if w:
                st.write(f"📍 **Location:** {w['name']} | **Temp:** {w['temp']}°C | **Humidity:** {w['hum']}%")
                sn, sp, sk = (100, 45, 35) if soil == "High Nitrogen" else (35, 75, 75) if soil == "P/K Rich" else (70, 45, 40)
                res = predict_crop(sn, sp, sk, w['temp'], w['hum'], 6.5, avg_rain)
                st.success(f"🌟 **Recommended Crop:** {res}")
                show_crop_details(res)

    # Tab 3: Soil Test CSV Processing
    with t3:
        up_file = st.file_uploader("Upload Soil CSV", type=["csv"])
        if up_file:
            df = pd.read_csv(up_file)
            df["Recommended_Crop"] = [predict_crop(r.get("N", 75), r.get("P", 45), r.get("K", 40), r.get("temperature", 23.0), r.get("humidity", 75.0), r.get("ph", 6.5), r.get("rainfall", 180.0)) for _, r in df.iterrows()]
            st.dataframe(df, use_container_width=True)
        else:
            sn, sp, sk = st.number_input("N", 75), st.number_input("P", 45), st.number_input("K", 40)
            if st.button("Evaluate Sample", key="b3"):
                res = predict_crop(sn, sp, sk, 23.0, 75.0, 6.5, 180.0)
                st.success(f"🌾 **Recommended Crop:** {res}")
                show_crop_details(res)

    # Tab 4: Plant.id Leaf Disease Assessment
    with t4:
        st.subheader("🍃 Leaf Disease Detection")
        api_key = st.text_input("Plant.id API Key:", type="password", help="Enter your Plant.id API key here")
        img_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
        
        if img_file:
            st.image(img_file, caption="Uploaded Leaf Image", width=250)
            
        if st.button("Run Diagnostic", key="b4"):
            if not api_key.strip():
                st.warning("⚠️ Please enter a valid Plant.id API Key above.")
            elif not img_file:
                st.warning("⚠️ Please upload a leaf image first.")
            else:
                with st.spinner("Analyzing leaf image via Plant.id API..."):
                    res = check_plant_disease_api(img_file.getvalue(), api_key.strip())
                    if res and res.status_code in (200, 201):
                        data = res.json().get("result", {})
                        diseases = data.get("disease", {}).get("suggestions", [])
                        if diseases:
                            st.success("Diagnostic Assessment Complete:")
                            for i, d in enumerate(diseases[:3]):
                                st.write(f"**{i+1}. {d.get('name')}** — `{d.get('probability', 0)*100:.1f}% Confidence`")
                        else:
                            st.info("No disease detected with high confidence.")
                    else:
                        st.error("API Connection Failed or Invalid API Key. Please check your credentials.")
                 
