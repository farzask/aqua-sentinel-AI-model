import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, db

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Initialize Firebase
# Make sure to place your firebase-adminsdk-*.json file in the project directory
try:
    cred = credentials.Certificate('aqua-sentinel-90685-firebase-adminsdk-fbsvc-bd6f9aeb57.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://aqua-sentinel-90685.firebaseio.com'
    })
    firebase_initialized = True
except Exception as e:
    print(f"Firebase initialization warning: {e}")
    firebase_initialized = False

# Global variables for model and scaler
ann_model = None
scaler = None
features = ['ph', 'Solids', 'Turbidity']
MODEL_FILE = 'ann_model.pkl'
SCALER_FILE = 'scaler.pkl'

def train_and_save_model():
    """Train the model and save it for later use"""
    global ann_model, scaler
    
    print("---------------------------------------------------")
    print("LOADING AND TRAINING MODEL")
    print("---------------------------------------------------")
    
    # Load Dataset
    df = pd.read_csv('dataset.csv')
    
    # Handle Missing Values
    imputer = SimpleImputer(strategy='mean')
    df['ph'] = imputer.fit_transform(df[['ph']])
    
    # Select features
    X = df[features]
    y = df['Potability']
    
    # Split: 80% for Training, 20% for Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create Scaled Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ANN Model
    ann_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    ann_model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_ann = ann_model.predict(X_test_scaled)
    acc_ann = accuracy_score(y_test, y_pred_ann)
    print(f"ANN Accuracy: {acc_ann:.4f}")
    
    # Save model and scaler
    joblib.dump(ann_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nModel saved to {MODEL_FILE}")
    print(f"Scaler saved to {SCALER_FILE}")
    
    return ann_model, scaler

def load_model():
    """Load pre-trained model and scaler"""
    global ann_model, scaler
    
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        ann_model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        print("Model and scaler loaded from files")
    else:
        ann_model, scaler = train_and_save_model()
    
    return ann_model, scaler

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make water potability predictions"""
    try:
        data = request.get_json()
        
        # Validate input
        if not all(key in data for key in features):
            return jsonify({'error': f'Missing required fields: {features}'}), 400
        
        # Extract values
        ph = float(data['ph'])
        solids = float(data['Solids'])
        turbidity = float(data['Turbidity'])
        
        # Prepare data for prediction
        input_data = np.array([[ph, solids, turbidity]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = ann_model.predict(input_scaled)[0]
        probability = ann_model.predict_proba(input_scaled)[0]
        
        # Prepare response
        result = {
            'potability': int(prediction),
            'potable': prediction == 0,
            'probability_potable': float(probability[0]),
            'probability_unsafe': float(probability[1]),
            'input': {
                'ph': ph,
                'Solids': solids,
                'Turbidity': turbidity
            }
        }
        
        # Save to Firebase if initialized
        if firebase_initialized:
            try:
                db.reference('predictions').push(result)
            except Exception as e:
                print(f"Firebase save warning: {e}")
        
        return jsonify(result), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'Model API is running', 'model_loaded': ann_model is not None}), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Artificial Neural Network (ANN)',
        'hidden_layers': [100, 50],
        'features': features,
        'input_scaling': 'StandardScaler'
    }), 200

if __name__ == '__main__':
    # Load or train model on startup
    load_model()
    
    # Run Flask app
    print("\n---------------------------------------------------")
    print("STARTING FLASK API SERVER")
    print("---------------------------------------------------")
    print("API running on http://localhost:5000")
    print("Use POST /predict to make predictions")
    print("---------------------------------------------------\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)