from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model safely
model = None
try:
    if os.path.exists('car_model_rf.pkl'):
        model = joblib.load('car_model_rf.pkl')
except Exception as e:
    print(f"Model load error: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # handle signup logic here or redirect to dashboard
        return render_template('dashboard.html')
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return render_template('dashboard.html')
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        year = int(data.get('year', 2020))
        kms_driven = float(data.get('kms_driven', 40000))
        fuel_type = data.get('fuel_type', 'Petrol')
        transmission = data.get('transmission', 'Manual')

        age = 2026 - year
        log_kms = np.log(kms_driven + 1)
        fuel_diesel = 1 if fuel_type == 'Diesel' else 0
        fuel_petrol = 1 if fuel_type == 'Petrol' else 0
        trans_auto = 1 if transmission == 'Automatic' else 0

        fair_price = 7.55
        if model is not None:
            try:
                input_features = np.array([[age, log_kms, fuel_diesel, fuel_petrol, trans_auto]])
                fair_price = float(model.predict(input_features)[0])
            except Exception as model_err:
                print(f"Model warning: {model_err}")
                fair_price = round(max(1.5, 12.0 - (age * 0.75) - (kms_driven / 25000.0)), 2)

        fair_price = round(max(0.5, fair_price), 2)
        low_est = round(fair_price * 0.93, 2)
        high_est = round(fair_price * 1.07, 2)

        return jsonify({
            'status': 'success',
            'predicted_price': fair_price,
            'Low_estimate': low_est,
            'high_estimate': high_est
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
