import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, db
import streamlit as st

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Global variables
# Model expects ['ph', 'Solids', 'Turbidity']; Firebase uses 'tds' for Solids
features = ['ph', 'Solids', 'Turbidity']
MODEL_FILE = 'ann_model.pkl'
SCALER_FILE = 'scaler.pkl'

# Module-level cache of latest sensor snapshot and prediction result.
# These persist across Streamlit reruns within the same server process.
_sensor_cache = {}   # mirrors the sensors node in Firebase
_last_result = {
    'ph': None,
    'tds': None,
    'turbidity': None,
    'potability': None,   # 1 = potable, 0 = not potable
    'timestamp': None,
    'error': None,
}

# ── Firebase initialisation ──────────────────────────────────────────────────
firebase_initialized = False
try:
    if not firebase_admin._apps:
        secret_dict = json.loads(st.secrets["firebase_service_account"])
        cred = credentials.Certificate(secret_dict)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://aqua-sentinel-90685-default-rtdb.firebaseio.com/'
        })
    firebase_initialized = True
except Exception as e:
    st.warning(f"Firebase initialization warning: {e}")


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
    """Returns (potability_int, probability_potable, probability_unsafe).
    potability: 1 = potable, 0 = not potable (matches Potability column in dataset).
    tds is mapped to the 'Solids' feature expected by the model.
    """
    input_data = np.array([[ph, tds, turbidity]])
    input_scaled = scaler.transform(input_data)

    prediction = int(ann_model.predict(input_scaled)[0])
    probability = ann_model.predict_proba(input_scaled)[0]
    return prediction, float(probability[1]), float(probability[0])


# ── Firebase real-time listener ──────────────────────────────────────────────
def _build_listener(ann_model, scaler):
    """Returns a callback suitable for db.reference('sensors').listen()."""

    def on_change(event):
        global _sensor_cache, _last_result

        path = event.path   # '/' on initial load, '/ph', '/tds', etc. on updates
        data = event.data

        # Ignore updates we wrote ourselves (potability)
        if path == '/potability':
            return

        # Merge incoming data into our local cache
        if path == '/':
            if isinstance(data, dict):
                _sensor_cache = dict(data)
            else:
                return
        else:
            key = path.lstrip('/')
            _sensor_cache[key] = data

        # Extract sensor values; skip if any are missing
        try:
            ph = float(_sensor_cache['ph'])
            tds = float(_sensor_cache['tds'])
            turbidity = float(_sensor_cache['turbidity'])
        except (KeyError, TypeError, ValueError):
            return

        # Predict
        try:

            if ph >= 14.0 and ph <= 1.0 or tds >= 1.0 and tds <= 52000.0 or turbidity <= 10.0 and turbidity >= 1.0:
                if ph >= 3 or ph <= 12 or turbidity > 5:
                    potability = 0
                    prob_potable = 0
                    prob_unsafe = 1
                else:
                    potability, prob_potable, prob_unsafe = run_prediction(
                ann_model, scaler, ph, tds, turbidity
                )

            
        except Exception as exc:
            _last_result['error'] = str(exc)
            return

        # Write only the potability key back to Firebase
        try:
            db.reference('sensors/potability').set(potability)
        except Exception as exc:
            _last_result['error'] = f"Firebase write error: {exc}"
            return

        # Update local result for the UI
        _last_result.update({
            'ph': ph,
            'tds': tds,
            'turbidity': turbidity,
            'potability': potability,
            'prob_potable': prob_potable,
            'prob_unsafe': prob_unsafe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': None,
        })

    return on_change


@st.cache_resource
def start_listener(_ann_model, _scaler):
    """Start the Firebase listener once per server process."""
    if not firebase_initialized:
        return None
    try:
        callback = _build_listener(_ann_model, _scaler)
        handle = db.reference('sensors').listen(callback)
        return handle
    except Exception as e:
        return None


# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("AquaSentinel - Water Potability Predictor")

ann_model, scaler = load_model()

# Start the real-time listener (no-op if already running)
start_listener(ann_model, scaler)

# ── Section 1: Live Firebase sensor display ──────────────────────────────────
st.header("Live Sensor Prediction (Firebase)")
st.caption(
    "The listener runs in the background. Whenever `ph`, `tds`, or `turbidity` "
    "changes in `sensors/`, the model predicts and writes `sensors/potability` (1 or 0)."
)

if st.button("Refresh"):
    st.rerun()

r = _last_result
if r['timestamp'] is None:
    st.info("Waiting for sensor data from Firebase…")
elif r['error']:
    st.error(r['error'])
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("pH", f"{r['ph']:.2f}")
    col2.metric("TDS (mg/L)", f"{r['tds']:.2f}")
    col3.metric("Turbidity (NTU)", f"{r['turbidity']:.2f}")

    if r['potability'] == 1:
        st.success(f"Water is POTABLE (Safe to drink)  —  `sensors/potability` set to **1**")
    else:
        st.error(f"Water is NOT POTABLE (Unsafe)  —  `sensors/potability` set to **0**")

    st.write(f"**Probability Safe:** {r['prob_potable']:.2%}")
    st.write(f"**Probability Unsafe:** {r['prob_unsafe']:.2%}")
    st.caption(f"Last updated: {r['timestamp']}")

st.divider()

# ── Section 2: Manual UI prediction (not saved to Firebase) ──────────────────
st.header("Manual Prediction (UI only)")
st.write("Results are displayed here only and are **not** saved to Firebase.")

with st.form("prediction_form"):
    ph_ui = st.number_input("pH", min_value=1.0, max_value=14.0, value=7.0, step=0.01)
    tds_ui = st.number_input("TDS / Solids (mg/L)", min_value=50.0, max_value=50000.0, value=1000.0, step=1.0)
    turbidity_ui = st.number_input("Turbidity (NTU)", min_value=1.0, max_value=10.0, value=10.0, step=0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    if ph_ui >= 14.0 and ph_ui <= 1.0 or tds_ui >= 1.0 and tds_ui <= 52000.0 or turbidity_ui <= 10.0 and turbidity_ui >= 1.0:
        if ph_ui >= 3 or ph_ui <= 12 or turbidity_ui > 5:
            potability = 0
            prob_potable = 0
            prob_unsafe = 1
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
