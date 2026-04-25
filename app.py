import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from typing import Any

import firebase_admin
from firebase_admin import credentials, db
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
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
            secret = st.secrets["firebase_service_account"]
            # secret = st.secrets["firebase"]
            cred_dict = json.loads(secret) if isinstance(secret, str) else json.loads(json.dumps(dict(secret)))
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://aqua-sentinel-90685-default-rtdb.firebaseio.com/'
            })
        return True
    except Exception as e:
        return str(e)


# ── Model helpers ────────────────────────────────────────────────────────────
def train_and_save_model():
    df = pd.read_csv('dataset.csv')

    # Impute NaN across all used features, not just ph
    imputer = SimpleImputer(strategy='mean')
    df[features] = imputer.fit_transform(df[features])

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

    # Balance classes so the model doesn't bias toward NOT POTABLE
    sample_weights = compute_sample_weight('balanced', y_train)
    ann_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)  # type: ignore[call-arg]

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
    print(f"[DEBUG MODEL] Raw model prediction: {prediction}, prob_unsafe={probability[0]:.4f}, prob_potable={probability[1]:.4f}")
    return prediction, float(probability[1]), float(probability[0])


# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("AquaSentinel - Water Potability Predictor")

firebase_result = init_firebase()
if firebase_result is not True:
    st.error(f"Firebase init failed: {firebase_result}")

ann_model, scaler = load_model()

# ── Firebase SSE listener (runs once, updates shared store on data change) ────
@st.cache_resource
def setup_sensor_listener(_firebase_ready):
    """Registers a Firebase SSE listener. Fires only when sensor data changes."""
    store: dict[str, Any] = {"data": None, "last_updated": None, "error": None}

    if _firebase_ready is not True:
        return store

    # Initial fetch so the UI has data immediately before the SSE connection is established
    try:
        initial = db.reference('sensors').get()
        if isinstance(initial, dict):
            store["data"] = initial
            store["last_updated"] = datetime.now()
            print(f"[DEBUG] Initial fetch: {initial}")
    except Exception as e:
        store["error"] = str(e)
        return store

    def on_change(event):
        try:
            if event.data is None:
                store["data"] = None
            elif isinstance(event.data, dict):
                # Full node update (e.g. initial push or whole sensors node replaced)
                store["data"] = event.data
                store["last_updated"] = datetime.now()
                store["error"] = None
                print(f"[DEBUG] Full update: {event.data}")
            else:
                # Partial update — a single field changed (e.g. /ph -> 7.2)
                field = event.path.lstrip('/')
                # Ignore writes we made ourselves to avoid a feedback loop
                if field == 'potability':
                    return
                if store["data"] is None:
                    store["data"] = {}
                if field:
                    store["data"][field] = event.data
                store["last_updated"] = datetime.now()
                store["error"] = None
                print(f"[DEBUG] Partial update: {field} = {event.data}")
        except Exception as e:
            store["error"] = str(e)
            print(f"[DEBUG] Listener error: {e}")

    try:
        db.reference('sensors').listen(on_change)
    except Exception as e:
        store["error"] = str(e)

    return store


# ── Section 1: Live Firebase sensor display ──────────────────────────────────
st.header("Live Sensor Prediction")

# Light poll just to re-render when listener has pushed new data (no Firebase call)
st_autorefresh(interval=1000, key="ui_refresh")

if st.button("Refresh"):
    st.rerun()

if firebase_result is not True:
    st.error("Firebase not connected. Cannot fetch sensor data.")
else:
    sensor_store = setup_sensor_listener(firebase_result)

    if sensor_store["error"]:
        st.error(f"Listener error: {sensor_store['error']}")
    else:
        data = sensor_store["data"]

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
                print(f"[DEBUG LIVE] Input values: ph={ph}, tds={tds}, turbidity={turbidity}")
                ph_valid = 0.0 <= ph <= 14.0
                tds_valid = 0.0 <= tds <= 52000.0
                turbidity_valid = 0.0 <= turbidity <= 10.0
                print(f"[DEBUG LIVE] Range checks: ph_valid={ph_valid} (must be 0-14), tds_valid={tds_valid} (must be 0-52000), turbidity_valid={turbidity_valid} (must be 0-10)")

                if not (ph_valid and tds_valid and turbidity_valid):
                    potability, prob_potable, prob_unsafe = 0, 0.0, 1.0
                    print(f"[DEBUG LIVE] Validation failed: ph_valid={ph_valid}, tds_valid={tds_valid}, turbidity_valid={turbidity_valid} -> potability={potability}")
                elif ph < 4 or ph > 12 or turbidity > 5 or tds == 0:
                    # Automatic fail for extreme conditions
                    potability, prob_potable, prob_unsafe = 0, 0.0, 1.0
                    reason = []
                    if ph < 4:
                        reason.append("pH too low (<4)")
                    if ph > 12:
                        reason.append("pH too high (>12)")
                    if turbidity > 5:
                        reason.append("Turbidity too high (>5)")
                    if tds == 0:
                        reason.append("TDS is 0 (not drinkable)")
                    print(f"[DEBUG LIVE] Automatic fail: {', '.join(reason)}")
                else:
                    print(f"[DEBUG LIVE] ✓ Passed validation, calling model...")
                    potability, prob_potable, prob_unsafe = run_prediction(
                        ann_model, scaler, ph, tds, turbidity
                    )
                    print(f"[DEBUG LIVE] ✓ Model returned: potability={potability}, prob_potable={prob_potable:.4f}, prob_unsafe={prob_unsafe:.4f}")

                # Write potability back to Firebase if it differs from what Firebase currently holds
                stored_potability = sensor_store["data"].get("potability")
                if stored_potability != potability:
                    try:
                        db.reference('sensors/potability').set(potability)
                        sensor_store["data"]["potability"] = potability
                        print(f"[DEBUG FIREBASE] Wrote potability={potability} to Firebase (was {stored_potability})")
                    except Exception as e:
                        st.warning(f"Could not write potability to Firebase: {e}")
                        print(f"[DEBUG FIREBASE] Error writing to Firebase: {e}")

                if potability == 1:
                    st.success("Water is POTABLE (Safe to drink)")
                else:
                    st.error("Water is NOT POTABLE (Unsafe)")

                if sensor_store["last_updated"]:
                    st.caption(f"Last updated: {sensor_store['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")

st.divider()

# ── Section 2: Manual UI prediction (not saved to Firebase) ──────────────────
st.header("Manual Prediction")

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
        print(f"[DEBUG MANUAL] Validation failed: ph_valid={ph_valid}, tds_valid={tds_valid}, turbidity_valid={turbidity_valid} -> potability={potability}")
        st.warning("Input values out of realistic range. Please check and try again.")
    elif ph_ui < 4 or ph_ui > 12 or turbidity_ui > 5 or tds_ui == 0:
        # Automatic fail for extreme conditions
        potability, prob_potable, prob_unsafe = 0, 0.0, 1.0
        reason = []
        if ph_ui < 4:
            reason.append("pH too low (<4)")
        if ph_ui > 12:
            reason.append("pH too high (>12)")
        if turbidity_ui > 5:
            reason.append("Turbidity too high (>5)")
        if tds_ui == 0:
            reason.append("TDS is 0 (not drinkable)")
        print(f"[DEBUG MANUAL] Automatic fail: {', '.join(reason)}")
    else:
        print(f"[DEBUG MANUAL] ✓ Passed pre-checks, calling model...")
        potability, prob_potable, prob_unsafe = run_prediction(
            ann_model, scaler, ph_ui, tds_ui, turbidity_ui
        )
        print(f"[DEBUG MANUAL] Prediction: ph={ph_ui}, tds={tds_ui}, turbidity={turbidity_ui} -> potability={potability}, prob_potable={prob_potable}, prob_unsafe={prob_unsafe}")

    st.session_state["manual_result"] = potability
    st.session_state["manual_result_time"] = datetime.now()
    print(f"[DEBUG MANUAL] Stored result: potability={potability}, timestamp={st.session_state['manual_result_time']}")

if "manual_result" in st.session_state:
    elapsed = (datetime.now() - st.session_state["manual_result_time"]).total_seconds()
    print(f"[DEBUG DISPLAY] Manual result exists: potability={st.session_state['manual_result']}, elapsed={elapsed:.2f}s")
    print(f"[DEBUG DISPLAY] Potability = {potability}, prob_potable={prob_potable:.4f}, prob_unsafe={prob_unsafe:.4f} {elapsed=:.2f}s")
    if elapsed <= 120:
        if potability == 1:
            print(f"[DEBUG DISPLAY] Showing POTABLE result")
            st.success("Water is POTABLE (Safe to drink)")
        else:
            print(f"[DEBUG DISPLAY] Showing NOT POTABLE result")
            st.error("Water is NOT POTABLE (Unsafe)")
    else:
        print(f"[DEBUG DISPLAY] Result expired (elapsed > 120s), clearing")
        del st.session_state["manual_result"]
        del st.session_state["manual_result_time"]

