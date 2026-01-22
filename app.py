# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Extreme Rainfall Forecasting System", layout="wide")

st.title("Extreme Rainfall Forecasting System")
st.markdown("**Case Study: Kuching, Miri, Sibu**")

# -----------------------------
# 1. Station Selection
# -----------------------------

station = st.selectbox(
    "Select Station",
    ["Kuching", "Miri", "Sibu"]
)

BASE_MODEL_DIR = {
    "Kuching": "models/kuching",
    "Miri": "models/miri",
    "Sibu": "models/sibu"
}

model_dir = BASE_MODEL_DIR[station]

# -----------------------------
# 2. Load Model & Scaler
# -----------------------------

@st.cache_resource
def load_model_and_scaler(model_dir):
    model = keras.models.load_model(os.path.join(model_dir, "tcn_risk_aware.keras"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.save"))

    with open(os.path.join(model_dir, "feature_cols.txt")) as f:
        feature_cols = [line.strip() for line in f.readlines()]

    return model, scaler, feature_cols

model, scaler, feature_cols = load_model_and_scaler(model_dir)

st.success(f"Loaded model for {station}")

# -----------------------------
# 3. Upload Recent Data
# -----------------------------

st.subheader("Upload Recent 30-Day Data (CSV)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df_recent = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(df_recent.tail())

    # Check columns
    missing = [c for c in feature_cols if c not in df_recent.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        # -----------------------------
        # 4. Build Sequence
        # -----------------------------

        SEQ_LEN = 30

        if len(df_recent) < SEQ_LEN:
            st.error("Need at least 30 days of data.")
        else:
            X_raw = df_recent[feature_cols].values[-SEQ_LEN:]

            # Scale
            X_scaled = scaler.transform(X_raw)

            X_input = X_scaled.reshape(1, SEQ_LEN, len(feature_cols))

            # -----------------------------
            # 5. Predict
            # -----------------------------

            if st.button("Predict Next-Day Rainfall"):
                y_pred_scaled = model.predict(X_input)[0, 0]

                # Inverse transform rainfall only
                tmp = np.zeros((1, len(feature_cols)))
                tmp[0, 0] = y_pred_scaled
                y_pred_mm = scaler.inverse_transform(tmp)[0, 0]

                st.subheader("Prediction Result")

                st.metric("Predicted Rainfall (mm)", f"{y_pred_mm:.2f}")

                # -----------------------------
                # 6. Risk Classification
                # -----------------------------

                def classify_risk(rain_mm):
                    if rain_mm < 40:
                        return "Normal", "🟢 No immediate flood risk."
                    elif rain_mm < 60:
                        return "Heavy Rain", "🟡 Monitor conditions. Potential localized flooding."
                    else:
                        return "Extreme Rainfall", "🔴 HIGH RISK: Flood alert. Preparedness required."

                risk_level, message = classify_risk(y_pred_mm)

                st.markdown(f"### Risk Level: **{risk_level}**")
                st.info(message)

                # -----------------------------
                # 7. Plot Last 30 Days + Forecast
                # -----------------------------

                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(df_recent["rainfall_mm"].values[-SEQ_LEN:], label="Last 30 Days")
                ax.scatter(SEQ_LEN, y_pred_mm, color="red", label="Forecast (t+1)")
                ax.set_title(f"{station} Rainfall Forecast")
                ax.set_xlabel("Days")
                ax.set_ylabel("Rainfall (mm)")
                ax.legend()

                st.pyplot(fig)

