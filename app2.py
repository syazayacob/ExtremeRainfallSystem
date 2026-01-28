# Extreme Rainfall Forecasting System (Model 2: Risk-Aware TCN)

# app.py

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tcn import TCN
from datetime import datetime

# Streamlit Configuration

st.set_page_config(page_title="Extreme Rainfall Forecasting System", layout="wide")

# Custom CSS to match the modern dashboard look
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; }
    .main { background-color: #f8f9fa; }
    </style>
    """, unsafe_allow_html=True) # Corrected parameter name

st.title("🌧️ Extreme Rainfall Forecasting System")
st.markdown("**Risk-Aware TCN | Case Study: Kuching, Miri, Sibu**")

# -----------------------------------------------------
# 1. Custom Loss Function (Risk-Aware)
# -----------------------------------------------------
# Penalizes underestimation of extreme rainfall

def asymmetric_mse(y_true, y_pred):
    diff = y_pred - y_true
    loss = tf.where(diff < 0, tf.square(diff) * 2.0, tf.square(diff))
    return tf.reduce_mean(loss)


# -----------------------------
# 2. Load Model & Scaler
# -----------------------------

@st.cache_resource
def load_model_and_scaler(model_dir):
    #model = keras.models.load_model(os.path.join(model_dir, "tcn_risk_aware.keras"))

    # Load trained TCN model, scaler, and feature column list.
    model = keras.models.load_model(
        os.path.join(model_dir, "tcn_risk_aware.keras"),
        custom_objects={
            "TCN": TCN,
            "asymmetric_mse": asymmetric_mse
            }
    )

    scaler = joblib.load(os.path.join(model_dir, "robust_scaler.save"))

    with open(os.path.join(model_dir, "feature_cols.txt"), "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]

    return model, scaler, feature_cols

# -----------------------------
# 3. Sidebar Station Selection 
# -----------------------------
with st.sidebar:
    st.header("Settings")
    station = st.selectbox("Select Rainfall Station", ["Kuching", "Miri", "Sibu"])
    
    MODEL_DIRS = {
        "Kuching": "models/kuching",
        "Miri": "models/miri",
        "Sibu": "models/sibu"
    }
    
    # Try loading files
    try:
        model, scaler, feature_cols = load_model_and_scaler(MODEL_DIRS[station])
        st.success(f"✅ {station} Model Loaded")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# -----------------------------
# 4. Upload Recent Data
# -----------------------------

st.subheader("📤 Upload Recent 30-Day Meteorological Data (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df_recent = pd.read_csv(uploaded_file)

    # Clean up column names (handles minor mismatches)
    df_recent.columns = df_recent.columns.str.strip().str.lower()
    df_recent = df_recent.rename(columns={
        "relative_humidity": "relative_humidity_percent",
        "pressure": "pressure_hpa"
    })

    st.write("Preview of uploaded data:")
    st.dataframe(df_recent.tail())

    # -------------------------------------------------
    # 5. Validate Input Columns
    # -------------------------------------------------

    # Define the threshold from your model training
    Q95_THRESHOLD = 47.73

    st.info(
        f"'is_extreme' feature generated internally using threshold {Q95_THRESHOLD} mm"
    )

    # The scaler expects 8 columns, but the CSV usually has 7 physical ones
    physical_features = [c for c in feature_cols if c != "is_extreme"]
    missing = [c for c in physical_features if c not in df_recent.columns]
    
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()
    
    # Create is_extreme feature (MODEL-EXPECTED)
    df_recent["is_extreme"] = (df_recent["rainfall_mm"] >= Q95_THRESHOLD).astype(int)

    
    # -------------------------------------------------
    # 6. UI Metrics Dashboard (Modern Style)
    # -------------------------------------------------

    last_day = df_recent.iloc[-1]
    prev_day = df_recent.iloc[-2]

    st.divider()
    st.subheader(f"📍 Current Conditions at {station}")
    
    # Grid Row 1
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rainfall", f"{last_day['rainfall_mm']} mm", 
              delta=f"{round(last_day['rainfall_mm'] - prev_day['rainfall_mm'], 1)} mm")
    m2.metric("Humidity", f"{last_day['relative_humidity_percent']}%", 
              delta=f"{round(last_day['relative_humidity_percent'] - prev_day['relative_humidity_percent'], 1)}%")
    m3.metric("Wind Speed", f"{last_day['wind_ms']} m/s", 
              delta=f"{round(last_day['wind_ms'] - prev_day['wind_ms'], 2)} m/s", delta_color="inverse")
    m4.metric("Pressure", f"{last_day['pressure_hpa']} hPa", 
              delta=f"{round(last_day['pressure_hpa'] - prev_day['pressure_hpa'], 1)} hPa", delta_color="normal")

    # Grid Row 2
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Wet Bulb Temp", f"{last_day['wet_bulb_c']}°C", 
              delta=f"{round(last_day['wet_bulb_c'] - prev_day['wet_bulb_c'], 1)}°C")
    m6.metric("Evaporation", f"{last_day['evaporation_mm']} mm", 
              delta=f"{round(last_day['evaporation_mm'] - prev_day['evaporation_mm'], 2)} mm")
    m7.metric("Cloud Cover", f"{int(last_day['cloud_cover_oktas'])} Oktas")
    m8.metric("Extreme Regime", "YES" if last_day['is_extreme'] == 1 else "NO")

    
    # -----------------------------
    # 7. Trend Visualization
    # -----------------------------
    st.subheader("📈 Temperature & Humidity Trend (30d)")
    st.area_chart(df_recent[['wet_bulb_c', 'relative_humidity_percent']].tail(30))
    

    # -----------------------------
    # 8. Prediction
    # -----------------------------
    SEQ_LEN = 30
    if len(df_recent) < SEQ_LEN:
        st.warning(f"Note: Using only available {len(df_recent)} days. 30 days is optimal.")
        # Pad with zeros if data is short to prevent crash
        pad_size = SEQ_LEN - len(df_recent)
        X_raw = np.pad(df_recent[feature_cols].values, ((pad_size, 0), (0, 0)), mode='edge')
    else:
        X_raw = df_recent[feature_cols].values[-SEQ_LEN:]

    if st.button("🔮 Generate Next-Day Forecast", type="primary"):
        # Scale and Predict
        X_scaled = scaler.transform(X_raw)
        X_input = X_scaled.reshape(1, SEQ_LEN, len(feature_cols))
        
        y_pred_scaled = model.predict(X_input)[0, 0]

        # Inverse scaling for rainfall (first column)
        tmp = np.zeros((1, len(feature_cols)))
        tmp[0, 0] = y_pred_scaled
        y_pred_mm = scaler.inverse_transform(tmp)[0, 0]
        
        # Risk Logic
        if y_pred_mm < 40:
            risk, color, msg = "NORMAL", "green", "🟢 No immediate flood risk."
        elif y_pred_mm < 60:
            risk, color, msg = "HEAVY RAIN", "orange", "🟡 Monitor conditions. Potential localized flooding."
        else:
            risk, color, msg = "EXTREME RAINFALL", "red", "🔴 HIGH RISK: Flood alert. Preparedness required."

        # Display Result
        st.divider()
        st.subheader("📊 Forecast Result")
        st.metric("Forecasted Rainfall", f"{y_pred_mm:.2f} mm")
        st.markdown(f"### 🚨 Risk Level: :{color}[{risk}]")
        st.info(msg)

        
        # -----------------------------
        # 9. Visualization: Plot Last 30 Days + Forecast
        # -----------------------------

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            df_recent["rainfall_mm"].values[-SEQ_LEN:],
            label="Observed (Last 30 Days)"
        )
        ax.scatter(
            SEQ_LEN,
            y_pred_mm,
            color="red",
            label="Forecast (t+1)"
        )

        ax.set_title(f"{station} – Rainfall Forecast")
        ax.set_xlabel("Days")
        ax.set_ylabel("Rainfall (mm)")
        ax.legend()

        st.pyplot(fig)

    # Footer
    st.divider()
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    