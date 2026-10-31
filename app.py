from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import requests  # Required if calling a Google Cloud Vertex AI / Cloud Run endpoint

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        
        # Extract exact form inputs matching your Cardekho training columns
        car_model = data.get('model', '')
        year = int(data.get('year', 2020))
        kms_driven = float(data.get('kms_driven', 40000))
        fuel_type = data.get('fuel_type', 'Petrol')
        transmission = data.get('transmission', 'Manual')

        # Compute exact transformations used in your training pipeline
        age = 2026 - year
        log_kms = np.log(kms_driven + 1)
        
        # Option A: If your model is hosted on Google Cloud Vertex AI / Cloud Run API endpoint
        # GCP_ENDPOINT_URL = "https://your-cloud-run-url.a.run.app/predict"
        # payload = {"model": car_model, "year": year, "kms_driven": kms_driven, "fuel_type": fuel_type, "transmission": transmission}
        # response = requests.post(GCP_ENDPOINT_URL, json=payload)
        # result = response.json()
        
        # Option B: Local/Render inference using your exact feature array matching your trained model
        fuel_diesel = 1 if fuel_type == 'Diesel' else 0
        fuel_petrol = 1 if fuel_type == 'Petrol' else 0
        trans_auto = 1 if transmission == 'Automatic' else 0

        # Load your trained model artifact
        model = joblib.load('car_model_rf.pkl')
        
        # Construct feature vector matching your training data columns
        input_features = np.array([[age, log_kms, fuel_diesel, fuel_petrol, trans_auto]])
        fair_price = float(model.predict(input_features)[0])

        fair_price = round(max(0.5, fair_price), 2)
        low_est = round(fair_price * 0.93, 2)
        high_est = round(fair_price * 1.07, 2)

        return jsonify({
            'status': 'success',
            'predicted_price': fair_price,
            'low_estimate': low_est,
            'high_estimate': high_est
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
