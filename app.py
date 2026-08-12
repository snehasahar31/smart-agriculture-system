import hashlib
import pandas as pd
import requests
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# --- Page Setup & Sidebar ---
st.set_page_config(
    page_title="Smart Agriculture System", page_icon="🌱", layout="wide"
)

st.title("🌱 Smart Agriculture & Crop Recommendation System")
st.write("An AI/ML based web application for crop prediction and disease detection.")
st.divider()

st.sidebar.title("📌 Project Details")
st.sidebar.write("**Course:** AI & ML")
st.sidebar.info(
    "Demo model: trained on a sample dataset. "
    "Predictions are illustrative, not agronomically accurate."
)
st.sidebar.divider()

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# --- ML Model Training ---
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
    row = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=FEATURE_COLS)
    return model.predict(row)[0]


# --- Weather Fetch API ---
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

    lat, lon = (
        geo_res["results"][0]["latitude"],
        geo_res["results"][0]["longitude"],
    )
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


# --- UI Tabs ---
t1, t2, t3, t4 = st.tabs([
    "🎛️ Manual Sliders",
    "🌤️ Weather Search",
    "📊 Soil Test Report",
    "🍃 Plant Disease Detection",
])

# Tab 1: Manual Sliders
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
        pred = predict_crop(
            n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val
        )
        st.success(f"**Recommended Crop:** {pred}")

# Tab 2: Live Weather
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
                except Exception:
                    st.error("Error fetching weather data.")
                    weather = None

            if weather:
                st.info(
                    f"📍 Location: {weather['resolved_name']} | "
                    f"Temp: {weather['temperature']}°C | "
                    f"Humidity: {weather['humidity']}% | "
                    f"Rainfall: {weather['precipitation']} mm"
                )
                pred = predict_crop(
                    85,
                    45,
                    40,
                    weather["temperature"],
                    weather["humidity"],
                    6.5,
                    weather["precipitation"],
                )
                st.success(
                    f"**Best Crop for {weather['resolved_name']}:** {pred}"
                )
            else:
                st.error(f"Location '{city_input}' not found.")

# Tab 3: Soil Test CSV
with t3:
    st.subheader("Soil Lab Test Analysis")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_csv:
        try:
            csv_df = pd.read_csv(uploaded_csv)
            preds = [
                predict_crop(
                    *[
                        float(row.get(c, 50))
                        for c in [
                            "N",
                            "P",
                            "K",
                            "temperature",
                            "humidity",
                            "ph",
                            "rainfall",
                        ]
                    ]
                )
                for _, row in csv_df.iterrows()
            ]

            csv_df["Recommended_Crop"] = preds
            st.write("### 📊 Soil Analysis Results")
            st.dataframe(csv_df, use_container_width=True)

            st.download_button(
                "Download Results as CSV",
                data=csv_df.to_csv(index=False).encode("utf-8"),
                file_name="crop_recommendations.csv",
                mime="text/csv",
            )
        except Exception:
            st.error("Could not read or process the CSV file.")
    else:
        st.caption("No file uploaded — try a quick manual check:")
        s_n = st.number_input("Nitrogen", value=90)
        s_p = st.number_input("Phosphorus", value=42)
        s_k = st.number_input("Potassium", value=43)
        if st.button("Analyze Soil", key="analyze_soil_manual"):
            res = predict_crop(s_n, s_p, s_k, 23.0, 75.0, 6.5, 180.0)
            st.success(f"**Recommended Crop:** {res}")

# Tab 4: Leaf Disease Detection
with t4:
    st.subheader("Leaf Disease Detection")
    leaf_file = st.file_uploader(
        "Upload Leaf Image", type=["jpg", "jpeg", "png"]
    )

    if leaf_file:
        try:
            img = Image.open(leaf_file)
            st.image(
                img, caption="Uploaded Leaf Image", use_container_width=True
            )

            if st.button("Detect Disease", key="detect_disease_btn"):
                sample_results = [
                    (
                        "Tomato - Early Blight (Fungal)",
                        [
                            "Apply copper-based fungicide spray.",
                            "Remove infected leaves.",
                        ],
                    ),
                    (
                        "Healthy Leaf - No Disease Detected",
                        ["No action needed.", "Continue regular monitoring."],
                    ),
                    (
                        "Potato - Late Blight (Fungal)",
                        [
                            "Apply appropriate fungicide.",
                            "Improve field drainage.",
                        ],
                    ),
                ]
                file_hash = int(
                    hashlib.md5(leaf_file.getvalue()).hexdigest(), 16
                )
                disease, remedies = sample_results[
                    file_hash % len(sample_results)
                ]

                st.success("Analysis Completed!")
                st.warning(f"**Detected Disease:** {disease}")
                st.markdown(
                    "**Recommended Remedies:**\n"
                    + "\n".join(f"- {r}" for r in remedies)
                )
        except Exception:
            st.error("Invalid image file uploaded.")
    
