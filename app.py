import pandas as pd
import numpy as np
import joblib
import os
import json

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
features = ['ph', 'Solids', 'Turbidity']
MODEL_FILE = 'ann_model.pkl'
SCALER_FILE = 'scaler.pkl'

# Initialize Firebase
firebase_initialized = False
try:
    if not firebase_admin._apps:
        secret_dict = json.loads(st.secrets["firebase_service_account"])
        cred = credentials.Certificate(secret_dict)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://aqua-sentinel-90685.firebaseio.com'
        })
    firebase_initialized = True
except Exception as e:
    st.warning(f"Firebase initialization warning: {e}")


def train_and_save_model():
    """Train the model and save it for later use"""
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
    """Load pre-trained model and scaler, training if not found"""
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        ann_model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    else:
        ann_model, scaler = train_and_save_model()
    return ann_model, scaler


def make_prediction(ann_model, scaler, ph, solids, turbidity):
    """Run prediction and optionally save to Firebase"""
    input_data = np.array([[ph, solids, turbidity]])
    input_scaled = scaler.transform(input_data)

    prediction = ann_model.predict(input_scaled)[0]
    probability = ann_model.predict_proba(input_scaled)[0]

    result = {
        'potability': int(prediction),
        'potable': bool(prediction == 0),
        'probability_potable': float(probability[0]),
        'probability_unsafe': float(probability[1]),
        'input': {'ph': ph, 'Solids': solids, 'Turbidity': turbidity}
    }

    if firebase_initialized:
        try:
            db.reference('predictions').push(result)
        except Exception as e:
            st.warning(f"Firebase save warning: {e}")

    return result


# --- Streamlit UI ---
st.title("AquaSentinel - Water Potability Predictor")
st.write("Enter water quality parameters to predict if the water is safe to drink.")

ann_model, scaler = load_model()

with st.form("prediction_form"):
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.01)
    solids = st.number_input("Solids (mg/L)", min_value=0.0, value=10000.0, step=1.0)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0, step=0.01)
    submitted = st.form_submit_button("Predict")

if submitted:
    result = make_prediction(ann_model, scaler, ph, solids, turbidity)

    if result['potable']:
        st.success(f"Water is POTABLE (Safe to drink)")
    else:
        st.error(f"Water is NOT POTABLE (Unsafe)")

    st.write(f"**Probability Safe:** {result['probability_potable']:.2%}")
    st.write(f"**Probability Unsafe:** {result['probability_unsafe']:.2%}")
