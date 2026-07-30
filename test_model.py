import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tcn import TCN


def asymmetric_mse(y_true, y_pred):
    diff = y_pred - y_true
    loss = tf.where(
        diff < 0,
        tf.square(diff) * 2.0,
        tf.square(diff)
    )
    return tf.reduce_mean(loss)


MODEL_DIR = "models/kuching"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tcn_risk_aware.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "robust_scaler.save"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_cols.txt"
)


print("Loading model...")

model = keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "TCN": TCN,
        "asymmetric_mse": asymmetric_mse
    },
    compile=False
)

print("Model loaded")
print(model.input_shape)


print("Loading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded")
print(
    scaler.n_features_in_
)


with open(FEATURE_PATH) as f:
    feature_cols = [
        line.strip()
        for line in f.readlines()
    ]

print("Features:")
print(feature_cols)


print("Loading CSV...")

df = pd.read_csv(
    "./sample_data/mock_kuching_30days_heavy.csv"
)


Q95_THRESHOLD = 47.73

df["is_extreme"] = (
    df["rainfall_mm"] >= Q95_THRESHOLD
).astype(int)


print(df[feature_cols].tail())


SEQ_LEN = 30


X_raw = df[feature_cols].values[-SEQ_LEN:]

print("X raw:")
print(X_raw.shape)


print("Scaling...")

X_scaled = scaler.transform(
    X_raw
)


print("Scaled:")
print(X_scaled.shape)


X_input = X_scaled.reshape(
    1,
    SEQ_LEN,
    len(feature_cols)
)


print("Model input:")
print(X_input.shape)


print("Predicting...")


prediction = model.predict(
    X_input
)


print("Prediction:")
print(prediction)

print("DONE")