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

# Streamlit Configuration

st.set_page_config(page_title="Extreme Rainfall Forecasting System", layout="wide")

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
# 2. Station Selection
# -----------------------------

station = st.selectbox(
    "Select Rainfall Station",
    ["Kuching", "Miri", "Sibu"]
)

MODEL_DIRS = {
    "Kuching": "models/kuching",
    "Miri": "models/miri",
    "Sibu": "models/sibu"
}

model_dir = MODEL_DIRS[station]

# -----------------------------
# 3. Load Model & Scaler
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

model, scaler, feature_cols = load_model_and_scaler(model_dir)

st.success(f"Model loaded successfully for **{station}**")

# -----------------------------
# 4. Upload Recent Data
# -----------------------------

st.subheader("📤 Upload Recent 30-Day Meteorological Data (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df_recent = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df_recent.tail())

    # -------------------------------------------------
    # 5. Validate Input Columns
    # -------------------------------------------------

    # Define the threshold from your model training
    Q95_THRESHOLD = 47.73

    # The scaler expects 8 columns, but the CSV usually has 7 physical ones
    physical_features = [c for c in feature_cols if c != "is_extreme"]

    missing = [c for c in feature_cols if c not in df_recent.columns]
    
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Create is_extreme feature (MODEL-EXPECTED)
    df_recent["is_extreme"] = (df_recent["rainfall_mm"] >= Q95_THRESHOLD).astype(int)

    st.info(
        f"'is_extreme' feature generated internally using threshold {Q95_THRESHOLD} mm"
    )
    
    # -----------------------------
    # 6. Build Sequence
    # -----------------------------

    SEQ_LEN = 30

    if len(df_recent) < SEQ_LEN:
        st.error("At least 30 consecutive days of data are required.")
        st.stop()

        
    # Now X_raw will have exactly 8 columns (7 physical + 1 'is_extreme')
    X_raw = df_recent[feature_cols].values[-SEQ_LEN:]

    # Scale
    X_scaled = scaler.transform(X_raw)

    X_input = X_scaled.reshape(1, SEQ_LEN, len(feature_cols))

    # -----------------------------
    # 7. Predict
    # -----------------------------

    if st.button("🔮 Predict Next-Day Rainfall"):
            
        y_pred_scaled = model.predict(X_input)[0, 0]

        # Inverse scaling (rainfall assumed as first feature)
        tmp = np.zeros((1, len(feature_cols)))
        tmp[0, 0] = y_pred_scaled
        y_pred_mm = scaler.inverse_transform(tmp)[0, 0]


        # -------------------------------------------------
        # 8. Display Results
        # -------------------------------------------------

        st.subheader("📊 Forecast Result")

        st.metric("Predicted Rainfall (mm)", f"{y_pred_mm:.2f}")

        # -----------------------------
        # 9. Risk Classification
        # -----------------------------

        def classify_risk(rain_mm):
            if rain_mm < 40:
                return "Normal", "🟢 No immediate flood risk."
            elif rain_mm < 60:
                return "Heavy Rain", "🟡 Monitor conditions. Potential localized flooding."
            else:
                return "Extreme Rainfall", "🔴 HIGH RISK: Flood alert. Preparedness required."

        risk_level, message = classify_risk(y_pred_mm)

        st.markdown(f"### 🚨 Risk Level: **{risk_level}**")
        st.info(message)

        # -----------------------------
        # 10. Visualization: Plot Last 30 Days + Forecast
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

