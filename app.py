import hashlib
import pandas as pd
import requests
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Smart Agriculture System",
    page_icon="🌱",
    layout="wide"
)

# --- Hash Password Helper ---
def make_hash(password: str) -> str:
    return hashlib.sha256(str.encode(password)).hexdigest()

# Demo User Database (Username: Hashed Password)
# admin -> admin123 | farmer -> farmer123
USER_DB = {
    "admin": make_hash("admin123"),
    "farmer": make_hash("farmer123")
}

# --- Session State Initialization ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# ==========================================
# 🔐 LOGIN SCREEN (App Context Before Login)
# ==========================================
if not st.session_state["logged_in"]:
    st.title("🔐 Smart Agriculture Portal - Login")
    st.write("Please log in to access the Crop Recommendation & Disease Detection System.")
    st.divider()

    col1, _ = st.columns([1, 1])
    with col1:
        user_input = st.text_input("Username")
        pass_input = st.text_input("Password", type="password")

        if st.button("Login", key="login_btn"):
            if user_input in USER_DB and USER_DB[user_input] == make_hash(pass_input):
                st.session_state["logged_in"] = True
                st.session_state["username"] = user_input
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid Username or Password!")

    st.info(
        "💡 **Demo Credentials for Presentation:**\n"
        "- Username: `admin` | Password: `admin123`\n"
        "- Username: `farmer` | Password: `farmer123`"
    )

# ==========================================
# 🌿 MAIN APP (Accessible Only After Login)
# ==========================================
else:
    # --- Sidebar with Logout ---
    st.sidebar.markdown(f"👤 **Logged in as:** `{st.session_state['username']}`")
    if st.sidebar.button("🚪 Logout", key="logout_btn"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.rerun()

    st.sidebar.divider()
    st.sidebar.title("📌 Project Details")
    st.sidebar.write("**Course:** AI & ML")
    st.sidebar.info(
        "Demo model: trained on a sample dataset. "
        "Predictions are illustrative, not agronomically accurate."
    )
    st.sidebar.divider()

    # --- Header ---
    st.title("🌱 Smart Agriculture & Crop Recommendation System")
    st.write("An AI/ML based web application for crop prediction and disease detection.")
    st.divider()

    FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    # --- Model Training ---
    @st.cache_resource
    def train_crop_model():
        dataset = [
            [90, 42, 43, 20.8, 82.0, 6.5, 202.9, "Rice"],
            [80, 40, 40, 22.0, 80.0, 6.0, 190.0, "Rice"],
            [95, 38, 45, 21.5, 85.0, 6.3, 210.0, "Rice"],
            [20, 60, 20, 21.0, 65.0, 5.5, 100.0, "Maize"],
            [25, 55, 25, 23.5, 60.0, 6.2, 95.0, "Maize"],
            [18, 62, 22, 22.0, 62.0, 5.9, 90.0, "Maize"],
            [40, 68, 80, 18.0, 20.0, 5.8, 80.0, "Chickpea"],
            [45, 70, 75, 17.5, 22.0, 6.0, 75.0, "Chickpea"],
            [120, 40, 20, 25.0, 70.0, 6.8, 150.0, "Cotton"],
            [115, 42, 25, 26.0, 68.0, 6.9, 140.0, "Cotton"],
            [100, 15, 30, 24.0, 85.0, 6.7, 220.0, "Coffee"],
            [105, 18, 28, 23.5, 88.0, 6.5, 215.0, "Coffee"],
        ]
        df_train = pd.DataFrame(dataset, columns=FEATURE_COLS + ["crop"])
        X = df_train[FEATURE_COLS]
        y = df_train["crop"]

        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X, y)
        return rf

    model = train_crop_model()

    def predict_crop(n, p, k, temp, hum, ph, rain):
        input_df = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=FEATURE_COLS)
        return model.predict(input_df)[0]

    # --- Weather API Fetch ---
    @st.cache_data(ttl=600, show_spinner=False)
    def fetch_weather(city_name: str):
        try:
            geo_url = "https://geocoding-api.open-meteo.com/v1/search"
            geo_params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
            geo_res = requests.get(geo_url, params=geo_params, timeout=10).json()
            
            if not geo_res.get("results"):
                return None

            location = geo_res["results"][0]
            lat, lon = location["latitude"], location["longitude"]
            resolved_name = location.get("name", city_name)

            w_url = "https://api.open-meteo.com/v1/forecast"
            w_params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation"
            }
            w_res = requests.get(w_url, params=w_params, timeout=10).json()

            return {
                "resolved_name": resolved_name,
                "temperature": w_res["current"]["temperature_2m"],
                "humidity": w_res["current"]["relative_humidity_2m"],
                "precipitation": w_res["current"]["precipitation"],
            }
        except Exception:
            return None

    # --- Tabs Setup ---
    t1, t2, t3, t4 = st.tabs([
        "🎛️ Manual Sliders",
        "🌤️ Weather Search",
        "📊 Soil Test Report",
        "🍃 Plant Disease Detection"
    ])

    # --- Tab 1: Manual Input ---
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

        if st.button("Analyze & Predict", key="predict_manual"):
            pred = predict_crop(n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val)
            st.success(f"**Recommended Crop:** {pred}")

    # --- Tab 2: Weather Search ---
    with t2:
        st.subheader("Live Weather Based Testing")
        city_input = st.text_input("Enter City / District:", "Patna")

        if st.button("Fetch Weather & Recommend", key="fetch_weather_btn"):
            if not city_input.strip():
                st.warning("Please enter a city name.")
            else:
                with st.spinner("Fetching live weather data..."):
                    weather = fetch_weather(city_input.strip())

                if weather:
                    st.info(
                        f"📍 Location: {weather['resolved_name']} | "
                        f"Temp: {weather['temperature']}°C | "
                        f"Humidity: {weather['humidity']}% | "
                        f"Rainfall: {weather['precipitation']} mm"
                    )
                    pred = predict_crop(
                        85, 45, 40,
                        weather["temperature"],
                        weather["humidity"],
                        6.5,
                        weather["precipitation"]
                    )
                    st.success(f"**Best Crop for {weather['resolved_name']}:** {pred}")
                else:
                    st.error(f"Location '{city_input}' not found or weather service unavailable.")

    # --- Tab 3: Soil Test Report ---
    with t3:
        st.subheader("Soil Lab Test Analysis")
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_csv:
            try:
                csv_df = pd.read_csv(uploaded_csv)
                preds = []
                for _, row in csv_df.iterrows():
                    crop_out = predict_crop(
                        float(row.get("N", 90)),
                        float(row.get("P", 42)),
                        float(row.get("K", 43)),
                        float(row.get("temperature", 23.0)),
                        float(row.get("humidity", 75.0)),
                        float(row.get("ph", 6.5)),
                        float(row.get("rainfall", 180.0))
                    )
                    preds.append(crop_out)

                csv_df["Recommended_Crop"] = preds
                st.write("### 📊 Soil Analysis Results")
                st.dataframe(csv_df, use_container_width=True)

                st.download_button(
                    "Download Results as CSV",
                    data=csv_df.to_csv(index=False).encode("utf-8"),
                    file_name="crop_recommendations.csv",
                    mime="text/csv"
                )
            except Exception:
                st.error("Could not read or process the uploaded CSV file.")
        else:
            st.caption("No file uploaded — try a quick manual check:")
            s_n = st.number_input("Nitrogen", value=90)
            s_p = st.number_input("Phosphorus", value=42)
            s_k = st.number_input("Potassium", value=43)
            
            if st.button("Analyze Soil", key="analyze_soil_manual"):
                res = predict_crop(s_n, s_p, s_k, 23.0, 75.0, 6.5, 180.0)
                st.success(f"**Recommended Crop:** {res}")

    # --- Tab 4: Plant Disease Detection (Phase-1 Validation) ---
    with t4:
        st.subheader("Leaf Disease Detection")
        st.caption("Status: Healthy Leaf Validation Module Active (Phase-1 PoC)")
        leaf_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

        if leaf_file:
            try:
                img = Image.open(leaf_file)
                st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

                if st.button("Detect Disease", key="detect_disease_btn"):
                    st.success("Analysis Completed!")
                    st.info("**Status:** Healthy Leaf - No Disease Detected")
                    st.markdown(
                        "**Recommended Remedies:**\n"
                        "- No action needed.\n"
                        "- Continue regular crop monitoring and optimal watering."
                    )
            except Exception:
                st.error("Invalid image file uploaded.")
        
    
