import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from PIL import Image
import hashlib

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Agriculture System", page_icon="ðŸŒ±", layout="wide"
)

st.title("ðŸŒ± Smart Agriculture & Crop Recommendation System")
st.write("An AI/ML based web application for crop prediction and disease detection.")
st.divider()

# Sidebar info
st.sidebar.title("ðŸ“Œ Project Details")
st.sidebar.write("**Course:** AI & ML")
st.sidebar.info(
    "âš ï¸ Demo model: trained on a very small sample dataset (7 rows). "
    "Predictions are illustrative, not agronomically accurate."
)
st.sidebar.divider()

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ------------------------------------------------------------------
# Model training (cached)
# ------------------------------------------------------------------
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
    """Run a single prediction using a properly-named DataFrame
    (avoids sklearn 'X does not have valid feature names' warnings)."""
    row = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=FEATURE_COLS)
    return model.predict(row)[0]


# ------------------------------------------------------------------
# Weather fetch (cached, with timeout + real error handling)
# ------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(city_name: str):
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_res = requests.get(
        geo_url,
        params={"name": city_name, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    ).json()

    if not geo_res.get("results"):
        return None

    lat = geo_res["results"][0]["latitude"]
    lon = geo_res["results"][0]["longitude"]
    resolved_name = geo_res["results"][0].get("name", city_name)

    w_url = "https://api.open-meteo.com/v1/forecast"
    w_res = requests.get(
        w_url,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation",
        },
        timeout=10,
    ).json()

    return {
        "resolved_name": resolved_name,
        "temperature": w_res["current"]["temperature_2m"],
        "humidity": w_res["current"]["relative_humidity_2m"],
        "precipitation": w_res["current"]["precipitation"],
    }


# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
t1, t2, t3, t4 = st.tabs([
    "ðŸŽ›ï¸ Manual Sliders",
    "ðŸŒ¤ï¸ Weather Test (City Search)",
    "ðŸ§ª Soil Test Report",
    "ðŸƒ Plant Disease Detection",
])

# ---------------- Tab 1: Sliders ----------------
with t1:
    st.subheader("Manual Parameter Testing")
    col1, col2 = st.columns(2)
    with col1:
        n_val = st.slider("Nitrogen (N)", 0, 140, 85)
        p_val = st.slider("Phosphorus (P)", 0, 145, 45)
        k_val = st.slider("Potassium (K)", 0, 205, 40)
        ph_val = st.slider("Soil pH Level", 0.0, 14.0, 6.5)
    with col2:
        temp_val = st.slider("Temperature (Â°C)", 10.0, 50.0, 23.0)
        hum_val = st.slider("Humidity (%)", 10.0, 100.0, 75.0)
        rain_val = st.slider("Rainfall (mm)", 20.0, 300.0, 180.0)

    if st.button("Analyze & Predict", key="predict_manual"):
        pred = predict_crop(n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val)
        st.success(f"ðŸŒ± **Recommended Crop:** {pred}")

# ---------------- Tab 2: Live Weather API ----------------
with t2:
    st.subheader("Live Weather Based Testing")
    city_input = st.text_input("Enter City / District:", "Patna")

    if st.button("Fetch Weather & Recommend", key="fetch_weather_btn"):
        if not city_input.strip():
            st.warning("Please enter a city name.")
        else:
            with st.spinner("Fetching live weather data..."):
                try:
                    weather = fetch_weather(city_input.strip())
                except requests.exceptions.Timeout:
                    st.error("â±ï¸ Weather service timed out. Please try again.")
                    weather = None
                except requests.exceptions.RequestException:
                    st.error("ðŸŒ Could not reach the weather service. Check your internet connection.")
                    weather = None
                except (KeyError, ValueError):
                    st.error("âš ï¸ Unexpected response from the weather service.")
                    weather = None

            if weather is None:
                if city_input:
                    st.error(f"Location '{city_input}' not found. Please check the spelling.")
            else:
                st.info(
                    f"ðŸ“ Location: {weather['resolved_name']} | "
                    f"ðŸŒ¡ï¸ Temp: {weather['temperature']}Â°C | "
                    f"ðŸ’§ Humidity: {weather['humidity']}% | "
                    f"ðŸŒ§ï¸ Rainfall: {weather['precipitation']} mm"
                )
                pred = predict_crop(
                    85, 45, 40,
                    weather["temperature"], weather["humidity"], 6.5, weather["precipitation"],
                )
                st.success(f"ðŸŒ± **Best Crop for {weather['resolved_name']}'s Weather:** {pred}")

# ---------------- Tab 3: CSV Processing ----------------
with t3:
    st.subheader("Soil Lab Test Analysis")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_csv:
        try:
            csv_df = pd.read_csv(uploaded_csv)
        except Exception:
            st.error("âŒ Could not read the CSV file. Please check the file format.")
            csv_df = None

        if csv_df is not None:
            if csv_df.empty:
                st.warning("The uploaded CSV is empty.")
            else:
                missing_cols = [c for c in FEATURE_COLS if c not in csv_df.columns]
                if missing_cols:
                    st.warning(
                        f"âš ï¸ Missing columns {missing_cols} â€” default values will be used for them."
                    )

                with st.spinner("Analyzing soil samples..."):
                    preds = []
                    for _, row in csv_df.iterrows():
                        try:
                            n = float(row.get("N", 90))
                            p = float(row.get("P", 42))
                            k = float(row.get("K", 43))
                            temp = float(row.get("temperature", 23.0))
                            hum = float(row.get("humidity", 75.0))
                            ph = float(row.get("ph", 6.5))
                            rain = float(row.get("rainfall", 180.0))
                            crop_out = predict_crop(n, p, k, temp, hum, ph, rain)
                        except (ValueError, TypeError):
                            crop_out = "Invalid data"
                        preds.append(crop_out)

                csv_df["Recommended_Crop"] = preds

                st.write("### ðŸ“Š Soil Analysis & Crop Predictions")
                st.dataframe(csv_df, use_container_width=True)
                st.success("âœ… Report Processed Successfully!")

                st.download_button(
                    "â¬‡ï¸ Download Results as CSV",
                    data=csv_df.to_csv(index=False).encode("utf-8"),
                    file_name="crop_recommendations.csv",
                    mime="text/csv",
                )

                st.write("### ðŸŒ± Crop Recommendations")
                for idx, row in csv_df.iterrows():
                    sample_id = row.get("Sample_ID", row.get("Location", f"Sample #{idx + 1}"))
                    crop_name = row["Recommended_Crop"]
                    st.success(f"ðŸ“ **{sample_id}** âž” Recommended Crop: **{crop_name}**")
    else:
        st.caption("No file uploaded â€” try a quick manual soil check instead:")
        s_n = st.number_input("Nitrogen", value=90)
        s_p = st.number_input("Phosphorus", value=42)
        s_k = st.number_input("Potassium", value=43)
        if st.button("Analyze Soil", key="analyze_soil_manual"):
            res = predict_crop(s_n, s_p, s_k, 23.0, 75.0, 6.5, 180.0)
            st.success(f"ðŸŒ± **Recommended Crop:** {res}")

# ---------------- Tab 4: Image Detection ----------------
with t4:
    st.subheader("ðŸƒ Leaf Disease Detection")
    st.caption(
        "âš ï¸ Demo mode: no trained vision model is connected yet, so results below are "
        "simulated for demonstration purposes only and should not be used for real "
        "agronomic decisions."
    )
    leaf_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if leaf_file:
        try:
            img = Image.open(leaf_file)
            img.verify()
            leaf_file.seek(0)
            img = Image.open(leaf_file)
            st.image(img, caption="Uploaded Leaf Image", use_container_width=True)
        except Exception:
            st.error("âŒ Could not open this image. Please upload a valid JPG/PNG file.")
            img = None

        if img is not None and st.button("Detect Disease", key="detect_disease_btn"):
            with st.spinner("ðŸ”„ Processing image..."):
                # Simulated result: deterministic per-image so the same file
                # always yields the same demo output (no real ML model here).
                sample_results = [
                    ("Tomato - Early Blight (Fungal)",
                     ["Apply copper-based fungicide spray.", "Remove infected leaves."]),
                    ("Healthy Leaf - No Disease Detected",
                     ["No action needed.", "Continue regular monitoring."]),
                    ("Potato - Late Blight (Fungal)",
                     ["Apply appropriate fungicide.", "Improve field drainage and airflow."]),
                    ("Leaf - Nutrient Deficiency (Suspected)",
                     ["Get a soil test done.", "Consider balanced NPK fertilization."]),
                ]
                file_hash = int(hashlib.md5(leaf_file.getvalue()).hexdigest(), 16)
                disease, remedies = sample_results[file_hash % len(sample_results)]

            st.success("âœ… **Analysis Completed!**")
            st.warning(f"âš ï¸ **Detected Disease:** {disease}")
            st.markdown("**ðŸ’¡ Recommended Remedy:**\n" + "\n".join(f"- {r}" for r in remedies))
            st.caption("Simulated output â€” connect a real image-classification model for production use.")
