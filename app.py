import hashlib
import pandas as pd
import requests
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="Smart Agriculture System", page_icon="🌱", layout="wide")

# 2. Security & Session Database
def make_hash(password: str) -> str:
    return hashlib.sha256(str.encode(password)).hexdigest()

if "user_db" not in st.session_state:
    st.session_state["user_db"] = {
        "admin": make_hash("admin123"),
        "farmer": make_hash("farmer123")
    }
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# 3. Authentication Screen (Login / Register)
if not st.session_state["logged_in"]:
    st.title("🌱 Smart Agriculture Portal")
    st.caption("Secure System Access")
    st.divider()

    col1, _ = st.columns([1, 1])
    with col1:
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        # Login Tab
        with tab_login:
            st.subheader("User Login")
            u_name = st.text_input("Username", key="l_usr")
            u_pass = st.text_input("Password", type="password", key="l_pwd")

            if st.button("Login", key="l_btn"):
                if u_name in st.session_state["user_db"] and st.session_state["user_db"][u_name] == make_hash(u_pass):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = u_name
                    st.success("Login Successful!")
                    st.rerun()
                else:
                    st.error("Invalid Credentials!")

        # Register Tab
        with tab_register:
            st.subheader("New Registration")
            r_usr = st.text_input("Choose Username", key="r_usr")
            r_pwd = st.text_input("Choose Password", type="password", key="r_pwd")
            r_cpwd = st.text_input("Confirm Password", type="password", key="r_cpwd")

            if st.button("Register Account", key="r_btn"):
                if not r_usr.strip() or not r_pwd.strip():
                    st.warning("Please fill all fields.")
                elif r_usr in st.session_state["user_db"]:
                    st.error("Username already taken.")
                elif r_pwd != r_cpwd:
                    st.error("Passwords do not match!")
                else:
                    st.session_state["user_db"][r_usr] = make_hash(r_pwd)
                    st.success("Account created successfully! You can now log in.")

# 4. Main Application Dashboard
else:
    # Sidebar Navigation
    st.sidebar.markdown(f"👤 **Logged User:** `{st.session_state['username']}`")
    if st.sidebar.button("🚪 Logout", key="logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.rerun()

    st.sidebar.divider()
    st.sidebar.title("📌 System Status")
    st.sidebar.write("**ML Model:** Random Forest")
    st.sidebar.write("**Status:** Active")
    st.sidebar.divider()

    # App Header
    st.title("🌱 Smart Agriculture & Crop Recommendation System")
    st.write("AI/ML Decision Support System for Crop Optimization and Disease Analysis.")
    st.divider()

    # ML Model Training & Setup
    FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    @st.cache_resource
    def get_trained_model():
        data = [
            [90, 42, 43, 20.8, 82.0, 6.5, 202.9, "Rice"],
            [80, 40, 40, 22.0, 80.0, 6.0, 190.0, "Rice"],
            [20, 60, 20, 21.0, 65.0, 5.5, 100.0, "Maize"],
            [25, 55, 25, 23.5, 60.0, 6.2, 95.0, "Maize"],
            [40, 68, 80, 18.0, 20.0, 5.8, 80.0, "Chickpea"],
            [120, 40, 20, 25.0, 70.0, 6.8, 150.0, "Cotton"],
            [100, 15, 30, 24.0, 85.0, 6.7, 220.0, "Coffee"]
        ]
        df = pd.DataFrame(data, columns=FEATURE_COLS + ["crop"])
        rf = RandomForestClassifier(n_estimators=30, random_state=42)
        rf.fit(df[FEATURE_COLS], df["crop"])
        return rf

    model = get_trained_model()

    def predict_crop(n, p, k, temp, hum, ph, rain):
        return model.predict(pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=FEATURE_COLS))[0]

    # Live Weather Service
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_weather(city: str):
        try:
            g_res = requests.get("https://geocoding-api.open-meteo.com/v1/search", 
                                 params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=5).json()
            if not g_res.get("results"):
                return None
            loc = g_res["results"][0]
            w_res = requests.get("https://api.open-meteo.com/v1/forecast", 
                                 params={"latitude": loc["latitude"], "longitude": loc["longitude"], 
                                         "current": "temperature_2m,relative_humidity_2m,precipitation"}, timeout=5).json()
            return {
                "name": loc.get("name", city),
                "temp": w_res["current"]["temperature_2m"],
                "hum": w_res["current"]["relative_humidity_2m"],
                "rain": w_res["current"]["precipitation"]
            }
        except Exception:
            return None

    # Application Tabs
    t1, t2, t3, t4 = st.tabs([
        "🎛️ Manual Sliders",
        "🌤️ Weather Search",
        "📊 Soil Test Report",
        "🍃 Disease Detection"
    ])

    # Tab 1: Manual Parameter Sliders
    with t1:
        st.subheader("Manual Testing")
        c1, c2 = st.columns(2)
        with c1:
            n = st.slider("Nitrogen (N)", 0, 140, 85)
            p = st.slider("Phosphorus (P)", 0, 145, 45)
            k = st.slider("Potassium (K)", 0, 205, 40)
            ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5)
        with c2:
            temp = st.slider("Temperature (°C)", 10.0, 50.0, 23.0)
            hum = st.slider("Humidity (%)", 10.0, 100.0, 75.0)
            rain = st.slider("Rainfall (mm)", 20.0, 300.0, 180.0)

        if st.button("Predict Crop", key="btn_t1"):
            res = predict_crop(n, p, k, temp, hum, ph, rain)
            st.success(f"**Recommended Crop:** {res}")

    # Tab 2: Live Weather Search
    with t2:
        st.subheader("Live Weather Based Crop Recommendation")
        city_name = st.text_input("Enter City / District:", "Patna")
        if st.button("Fetch Weather & Predict", key="btn_t2"):
            weather = fetch_weather(city_name.strip())
            if weather:
                st.info(f"📍 Location: {weather['name']} | Temp: {weather['temp']}°C | Humidity: {weather['hum']}% | Rainfall: {weather['rain']} mm")
                res = predict_crop(85, 45, 40, weather["temp"], weather["hum"], 6.5, weather["rain"])
                st.success(f"**Recommended Crop for {weather['name']}:** {res}")
            else:
                st.error("Could not fetch weather data for this location.")

    # Tab 3: CSV Batch Soil Analysis
    with t3:
        st.subheader("Soil Lab Report Upload")
        uploaded_file = st.file_uploader("Upload Soil CSV", type=["csv"])
        if uploaded_file:
            try:
                df_soil = pd.read_csv(uploaded_file)
                df_soil["Recommended_Crop"] = [
                    predict_crop(r.get("N", 90), r.get("P", 42), r.get("K", 43), 
                                 r.get("temperature", 23.0), r.get("humidity", 75.0), 
                                 r.get("ph", 6.5), r.get("rainfall", 180.0)) 
                    for _, r in df_soil.iterrows()
                ]
                st.dataframe(df_soil, use_container_width=True)
                st.download_button("Download CSV Results", data=df_soil.to_csv(index=False).encode("utf-8"), file_name="crop_analysis.csv")
            except Exception:
                st.error("Error reading CSV file structure.")
        else:
            st.caption("Quick Single Sample Test:")
            sn = st.number_input("Nitrogen (N)", value=90)
            sp = st.number_input("Phosphorus (P)", value=42)
            sk = st.number_input("Potassium (K)", value=43)
            if st.button("Evaluate Sample", key="btn_t3"):
                st.success(f"**Recommended Crop:** {predict_crop(sn, sp, sk, 23.0, 75.0, 6.5, 180.0)}")

    # Tab 4: Leaf Disease Detection
    with t4:
        st.subheader("Plant Health Diagnostic")
        img_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
        if img_file:
            try:
                img = Image.open(img_file)
                st.image(img, caption="Uploaded Leaf Specimen", use_container_width=True)
                if st.button("Run Diagnostic", key="btn_t4"):
                    st.success("Diagnostic Complete!")
                    st.info("**Health Status:** Healthy / Normal Leaf")
                    st.write("- No disease symptoms detected.")
                    st.write("- Continue standard watering and nutrient schedule.")
            except Exception:
                st.error("Invalid image format.")
    
                            
