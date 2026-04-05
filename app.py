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
    print("[DEBUG] Attempting Firebase initialization...")
    if not firebase_admin._apps:
        secret_dict = json.loads(st.secrets["firebase_service_account"])
        cred = credentials.Certificate(secret_dict)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://aqua-sentinel-90685-default-rtdb.firebaseio.com/'
        })
        print("[DEBUG] Firebase initialized successfully")
    firebase_initialized = True
except Exception as e:
    print(f"[DEBUG] Firebase initialization error: {e}")
    st.warning(f"Firebase initialization warning: {e}")


# ── Model helpers ────────────────────────────────────────────────────────────
def train_and_save_model():
    print("[DEBUG] Starting model training...")
    df = pd.read_csv('dataset.csv')
    print(f"[DEBUG] Loaded dataset with shape: {df.shape}")

    imputer = SimpleImputer(strategy='mean')
    df['ph'] = imputer.fit_transform(df[['ph']])
    print("[DEBUG] Imputed missing pH values")

    X = df[features]
    y = df['Potability']
    print(f"[DEBUG] Features shape: {X.shape}, Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[DEBUG] Train/test split - Train: {X_train.shape}, Test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("[DEBUG] Data scaled")

    ann_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    ann_model.fit(X_train_scaled, y_train)
    print("[DEBUG] Model training completed")

    y_pred = ann_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"[DEBUG] Model accuracy: {acc:.4f}")
    st.info(f"Model trained. ANN Accuracy: {acc:.4f}")

    joblib.dump(ann_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"[DEBUG] Model and scaler saved to {MODEL_FILE} and {SCALER_FILE}")
    return ann_model, scaler


@st.cache_resource
def load_model():
    print("[DEBUG] Loading model...")
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        print(f"[DEBUG] Loading cached model from {MODEL_FILE}")
        return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)
    print("[DEBUG] Model files not found, training new model")
    return train_and_save_model()


def run_prediction(ann_model, scaler, ph, tds, turbidity):
    """Returns (potability_int, probability_potable, probability_unsafe).
    potability: 1 = potable, 0 = not potable (matches Potability column in dataset).
    tds is mapped to the 'Solids' feature expected by the model.
    """
    print(f"[DEBUG] Running prediction with pH={ph}, TDS={tds}, Turbidity={turbidity}")
    input_data = pd.DataFrame([[ph, tds, turbidity]], columns=features)
    input_scaled = scaler.transform(input_data)

    prediction = int(ann_model.predict(input_scaled)[0])
    probability = ann_model.predict_proba(input_scaled)[0]
    print(f"[DEBUG] Prediction result: {prediction}, Probabilities: Safe={probability[1]:.4f}, Unsafe={probability[0]:.4f}")
    return prediction, float(probability[1]), float(probability[0])


# ── Firebase real-time listener ──────────────────────────────────────────────
def _build_listener(ann_model, scaler):
    """Returns a callback suitable for db.reference('sensors').listen()."""

    def on_change(event):
        global _sensor_cache, _last_result

        path = event.path   # '/' on initial load, '/ph', '/tds', etc. on updates
        data = event.data
        print(f"[DEBUG] Firebase listener triggered - Path: {path}, Data: {data}")

        # Ignore updates we wrote ourselves (potability)
        if path == '/potability':
            print("[DEBUG] Ignoring potability update (self-written)")
            return

        # Merge incoming data into our local cache
        if path == '/':
            if isinstance(data, dict):
                _sensor_cache = dict(data)
                print(f"[DEBUG] Updated sensor cache: {_sensor_cache}")
            else:
                print("[DEBUG] Initial data is not a dict, skipping")
                return
        else:
            key = path.lstrip('/')
            _sensor_cache[key] = data
            print(f"[DEBUG] Updated sensor cache - {key}: {data}")

        # Extract sensor values; skip if any are missing
        try:
            ph = float(_sensor_cache['ph'])
            tds = float(_sensor_cache['tds'])
            turbidity = float(_sensor_cache['turbidity'])
            print(f"[DEBUG] Extracted sensor values - pH: {ph}, TDS: {tds}, Turbidity: {turbidity}")
        except (KeyError, TypeError, ValueError) as e:
            print(f"[DEBUG] Missing or invalid sensor data: {e}")
            return

        # Predict
        try:
            print("[DEBUG] Checking validation rules...")
            if ph >= 14.0 and ph <= 1.0 or tds >= 1.0 and tds <= 52000.0 or turbidity <= 10.0 and turbidity >= 1.0:
                print("[DEBUG] Input values within acceptable range")
                if ph <= 3 or ph >= 12 or turbidity > 5:
                    print("[DEBUG] Failed quality checks - setting potability to 0")
                    potability = 0
                    prob_potable = 0
                    prob_unsafe = 1
                else:
                    print("[DEBUG] Passed quality checks - running ANN prediction")
                    potability, prob_potable, prob_unsafe = run_prediction(
                ann_model, scaler, ph, tds, turbidity
                )

            
        except Exception as exc:
            print(f"[DEBUG] Prediction error: {exc}")
            _last_result['error'] = str(exc)
            return

        # Write only the potability key back to Firebase
        try:
            print(f"[DEBUG] Writing potability={potability} to Firebase")
            db.reference('sensors/potability').set(potability)
            print("[DEBUG] Successfully wrote to Firebase")
        except Exception as exc:
            print(f"[DEBUG] Firebase write error: {exc}")
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
        print(f"[DEBUG] Updated UI result cache - Potability: {potability}")

    return on_change


@st.cache_resource
def start_listener(_ann_model, _scaler):
    """Start the Firebase listener once per server process."""
    print("[DEBUG] Starting Firebase listener...")
    if not firebase_initialized:
        print("[DEBUG] Firebase not initialized, cannot start listener")
        return None
    try:
        callback = _build_listener(_ann_model, _scaler)
        handle = db.reference('sensors').listen(callback)
        print("[DEBUG] Firebase listener started successfully")
        return handle
    except Exception as e:
        print(f"[DEBUG] Error starting listener: {e}")
        return None


# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("AquaSentinel - Water Potability Predictor")

ann_model, scaler = load_model()

# Start the real-time listener (no-op if already running)
start_listener(ann_model, scaler)

# ── Section 1: Live Firebase sensor display ──────────────────────────────────
st.header("Live Sensor Prediction (Firebase)")

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

with st.form("prediction_form"):
    ph_ui = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.01)
    tds_ui = st.number_input("TDS / Solids (mg/L)", min_value=0.0, max_value=50000.0, value=1000.0, step=1.0)
    turbidity_ui = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=10.0, value=10.0, step=0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    print(f"[DEBUG] Manual prediction submitted - pH: {ph_ui}, TDS: {tds_ui}, Turbidity: {turbidity_ui}")
    if ph_ui >= 14.0 and ph_ui <= 1.0 or tds_ui >= 1.0 and tds_ui <= 52000.0 or turbidity_ui <= 10.0 and turbidity_ui >= 1.0:
        if ph_ui <= 3 or ph_ui >= 12 or turbidity_ui > 5:
            print("[DEBUG] Manual prediction: Failed quality checks")
            potability = 0
            prob_potable = 0
            prob_unsafe = 1
        else:
            print("[DEBUG] Manual prediction: Passed quality checks, running ANN")
            potability, prob_potable, prob_unsafe = run_prediction(
                ann_model, scaler, ph_ui, tds_ui, turbidity_ui
            )

    print(f"[DEBUG] Manual prediction result: {potability}")
    if potability == 1:
        st.success("Water is POTABLE (Safe to drink)")
    else:
        st.error("Water is NOT POTABLE (Unsafe)")
    st.write(f"**Probability Safe:** {prob_potable:.2%}")
    st.write(f"**Probability Unsafe:** {prob_unsafe:.2%}")
