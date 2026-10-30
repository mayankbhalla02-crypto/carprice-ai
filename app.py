import os
import sqlite3
from datetime import datetime
from functools import wraps
from flask import Flask, g, redirect, render_template, request, session, jsonify, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import ml_model
import os

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        year = int(data.get('year', 2020))
        kms_driven = float(data.get('kms_driven', 40000))
        fuel_type = data.get('fuel_type', 'Petrol')
        transmission = data.get('transmission', 'Manual')
        target_price = float(data.get('target_price', 0))

        age = 2026 - year
        log_kms = np.log(kms_driven + 1)
        fuel_diesel = 1 if fuel_type == 'Diesel' else 0
        fuel_petrol = 1 if fuel_type == 'Petrol' else 0
        trans_auto = 1 if transmission == 'Automatic' else 0

        fair_price = 7.55 # Safe default / fallback estimation
        if model is not None:
            try:
                # Try matching model input shape
                input_features = np.array([[age, log_kms, fuel_diesel, fuel_petrol, trans_auto]])
                fair_price = float(model.predict(input_features)[0])
            except Exception as model_err:
                print(f"Model prediction warning: {model_err}")
                # Fallback calculation if feature dimensions differ
                fair_price = round(max(1.5, 12.0 - (age * 0.75) - (kms_driven / 25000.0)), 2)

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
        # Always return JSON even on error so frontend doesn't crash with 500 HTML
        return jsonify({'status': 'error', 'message': str(e)}), 400

# Use /tmp for SQLite on cloud platforms like Render, or local directory for testing
if os.environ.get('RENDER'):
    DB_PATH = '/tmp/carusers.db'
else:
    DB_PATH = 'carusers.db'



DB_PATH = "carusers.db"
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your model here
# model = joblib.load('car_model_rf.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # your prediction logic here
    pass

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            org_name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            car_model TEXT,
            fuel TEXT,
            transmission TEXT,
            result TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    """)
    # Seed demo user
    existing = db.execute("SELECT 1 FROM users WHERE username = ?", ("demo@carai.com",)).fetchone()
    if not existing:
        db.execute(
            "INSERT INTO users (username, password_hash, org_name) VALUES (?, ?, ?)",
            ("demo@carai.com", generate_password_hash("demo123", method="pbkdf2:sha256"), "CarAI Expert Evaluator")
        )
    db.commit()
    db.close()

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

@app.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html", error=None)
    
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    
    if user is None or not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="Invalid email or password."), 401
        
    session["user_id"] = user["id"]
    session["org_name"] = user["org_name"]
    return redirect(url_for("dashboard"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html", error=None)
        
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    org_name = request.form.get("org_name", "").strip() or "Car Enthusiast"
    
    if "@" not in email or len(password) < 6:
        return render_template("signup.html", error="Provide a valid email and password (min 6 chars)."), 400
        
    db = get_db()
    existing = db.execute("SELECT 1 FROM users WHERE username = ?", (email,)).fetchone()
    if existing:
        return render_template("signup.html", error="Account already exists. Please log in."), 400
        
    db.execute(
        "INSERT INTO users (username, password_hash, org_name) VALUES (?, ?, ?)",
        (email, generate_password_hash(password, method="pbkdf2:sha256"), org_name)
    )
    db.commit()
    
    user = db.execute("SELECT * FROM users WHERE username = ?", (email,)).fetchone()
    session["user_id"] = user["id"]
    session["org_name"] = user["org_name"]
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", org_name=session.get("org_name", "User"))

@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    prediction = ml_model.predict_price(data)
    
    db = get_db()
    db.execute(
        """INSERT INTO predictions (user_id, car_model, fuel, transmission, result, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            session["user_id"],
            data.get("model", "Unknown"),
            data.get("fuel", "Petrol"),
            data.get("Type", "Manual"),
            prediction["estimated_range"],
            datetime.now().strftime("%d %b, %I:%M %p")
        )
    )
    db.commit()
    return jsonify(prediction)

if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
