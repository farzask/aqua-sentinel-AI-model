import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Model expects ['ph', 'Solids', 'Turbidity']; Firebase uses 'tds' for Solids
features = ['ph', 'Solids', 'Turbidity']
MODEL_FILE = 'ann_model.pkl'
SCALER_FILE = 'scaler.pkl'


# ── Firebase initialisation ──────────────────────────────────────────────────
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            with open('aqua-sentinel-90685-firebase-adminsdk-fbsvc-bd6f9aeb57.json') as f:
                secret_dict = json.load(f)
            cred = credentials.Certificate(secret_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://aqua-sentinel-90685-default-rtdb.firebaseio.com/'
            })
        return True
    except Exception as e:
        return str(e)


# ── Model helpers ────────────────────────────────────────────────────────────
def train_and_save_model():
    df = pd.read_csv('dataset.csv')
    imputer = SimpleImputer(strategy='mean')
    df['ph'] = imputer.fit_transform(df[['ph']])

    X = df[features]
    y = df['Potability']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ann_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    ann_model.fit(X_train_scaled, y_train)

    y_pred = ann_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.info(f"Model trained. ANN Accuracy: {acc:.4f}")

    joblib.dump(ann_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return ann_model, scaler


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)
    return train_and_save_model()


def run_prediction(ann_model, scaler, ph, tds, turbidity):
    input_data = pd.DataFrame([[ph, tds, turbidity]], columns=features)
    input_scaled = scaler.transform(input_data)
    prediction = int(ann_model.predict(input_scaled)[0])
    probability = ann_model.predict_proba(input_scaled)[0]
    return prediction, float(probability[1]), float(probability[0])


# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("AquaSentinel - Water Potability Predictor")

firebase_result = init_firebase()
if firebase_result is not True:
    st.error(f"Firebase init failed: {firebase_result}")

ann_model, scaler = load_model()

# ── Section 1: Live Firebase sensor display ──────────────────────────────────
st.header("Live Sensor Prediction (Firebase)")

# Auto-refresh every 3 seconds — on each rerun we read Firebase directly
st_autorefresh(interval=3000, key="firebase_autorefresh")

if st.button("Refresh"):
    st.rerun()

if firebase_result is not True:
    st.error("Firebase not connected. Cannot fetch sensor data.")
else:
    try:
        data = db.reference('sensors').get()
        print(f"[DEBUG] Fetched from Firebase: {data}")

        if not data or not isinstance(data, dict):
            st.info("Waiting for sensor data from Firebase…")
        else:
            ph = data.get('ph')
            tds = data.get('tds')
            turbidity = data.get('turbidity')

            if ph is None or tds is None or turbidity is None:
                st.warning(f"Incomplete sensor data: {data}")
            else:
                ph = float(ph)
                tds = float(tds)
                turbidity = float(turbidity)

                col1, col2, col3 = st.columns(3)
                col1.metric("pH", f"{ph:.2f}")
                col2.metric("TDS (mg/L)", f"{tds:.2f}")
                col3.metric("Turbidity (NTU)", f"{turbidity:.2f}")

                # Validation + prediction
                ph_valid = 0.0 <= ph <= 14.0
                tds_valid = 0.0 <= tds <= 52000.0
                turbidity_valid = 0.0 <= turbidity <= 10.0

                if not (ph_valid and tds_valid and turbidity_valid):
                    potability, prob_potable, prob_unsafe = 0, 0.0, 1.0
                elif ph <= 3 or ph >= 12 or turbidity > 5:
                    potability, prob_potable, prob_unsafe = 0, 0.0, 1.0
                else:
                    potability, prob_potable, prob_unsafe = run_prediction(
                        ann_model, scaler, ph, tds, turbidity
                    )

                # Write potability back to Firebase
                try:
                    db.reference('sensors/potability').set(potability)
                except Exception as e:
                    st.warning(f"Could not write potability to Firebase: {e}")

                if potability == 1:
                    st.success("Water is POTABLE (Safe to drink)  —  `sensors/potability` set to **1**")
                else:
                    st.error("Water is NOT POTABLE (Unsafe)  —  `sensors/potability` set to **0**")

                st.write(f"**Probability Safe:** {prob_potable:.2%}")
                st.write(f"**Probability Unsafe:** {prob_unsafe:.2%}")
                st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"Error reading from Firebase: {e}")
        print(f"[DEBUG] Firebase read error: {e}")

st.divider()

# ── Section 2: Manual UI prediction (not saved to Firebase) ──────────────────
st.header("Manual Prediction (UI only)")

with st.form("prediction_form"):
    ph_ui = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.01)
    tds_ui = st.number_input("TDS / Solids (mg/L)", min_value=0.0, max_value=50000.0, value=1000.0, step=1.0)
    turbidity_ui = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=10.0, step=0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    ph_valid = 0.0 <= ph_ui <= 14.0
    tds_valid = 0.0 <= tds_ui <= 52000.0
    turbidity_valid = 0.0 <= turbidity_ui <= 10.0

    if not (ph_valid and tds_valid and turbidity_valid):
        potability, prob_potable, prob_unsafe = 0, 0.0, 1.0
    elif ph_ui <= 3 or ph_ui >= 12 or turbidity_ui > 5:
        potability, prob_potable, prob_unsafe = 0, 0.0, 1.0
    else:
        potability, prob_potable, prob_unsafe = run_prediction(
            ann_model, scaler, ph_ui, tds_ui, turbidity_ui
        )

    if potability == 1:
        st.success("Water is POTABLE (Safe to drink)")
    else:
        st.error("Water is NOT POTABLE (Unsafe)")
    st.write(f"**Probability Safe:** {prob_potable:.2%}")
    st.write(f"**Probability Unsafe:** {prob_unsafe:.2%}")
