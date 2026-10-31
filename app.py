import logging
import os
import secrets
import socket
import sqlite3
import threading
import webbrowser
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (Flask, g, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import ml_model

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "carusers.db"
SECRET_PATH = BASE_DIR / ".secret_key"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("carai")


def load_secret_key():
    """Use FLASK_SECRET_KEY if set, otherwise create one once and reuse it."""
    env = os.environ.get("FLASK_SECRET_KEY")
    if env:
        return env
    try:
        if SECRET_PATH.exists():
            return SECRET_PATH.read_text().strip()
        key = secrets.token_hex(32)
        SECRET_PATH.write_text(key)
        return key
    except OSError:
        return secrets.token_hex(32)  # read-only folder: sessions last until restart


app = Flask(__name__)
app.secret_key = load_secret_key()
app.config.update(SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_HTTPONLY=True,
                  SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")))


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
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
    existing = db.execute("SELECT 1 FROM users WHERE username = ?", ("demo@carai.com",)).fetchone()
    if not existing:
        db.execute(
            "INSERT INTO users (username, password_hash, org_name) VALUES (?, ?, ?)",
            ("demo@carai.com", generate_password_hash("demo123", method="pbkdf2:sha256"),
             "CarAI Expert Evaluator"),
        )
    db.commit()
    db.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/"):
                return jsonify(error="Your session has ended. Sign in again."), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return redirect(url_for("dashboard" if session.get("user_id") else "login"))


def _auth_context():
    try:
        return {"bands": ml_model.band_info(), "car_count": len(ml_model.known_cars())}
    except Exception:
        log.exception("Model unavailable while rendering a sign-in page")
        return {"bands": [], "car_count": 0}


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if session.get("user_id"):
            return redirect(url_for("dashboard"))
        return render_template("login.html", error=None, **_auth_context())

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    user = get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if user is None or not check_password_hash(user["password_hash"], password):
        return render_template("login.html", error="Invalid email or password.",
                               **_auth_context()), 401

    session["user_id"] = user["id"]
    session["org_name"] = user["org_name"]
    return redirect(url_for("dashboard"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html", error=None, **_auth_context())

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    org_name = request.form.get("org_name", "").strip() or "Car Enthusiast"

    if "@" not in email or len(password) < 6:
        return render_template(
            "signup.html", error="Enter a valid email and a password of at least 6 characters.",
            **_auth_context()), 400

    db = get_db()
    if db.execute("SELECT 1 FROM users WHERE username = ?", (email,)).fetchone():
        return render_template("signup.html", error="That email already has an account. Sign in instead.",
                               **_auth_context()), 400

    db.execute(
        "INSERT INTO users (username, password_hash, org_name) VALUES (?, ?, ?)",
        (email, generate_password_hash(password, method="pbkdf2:sha256"), org_name),
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
    return render_template(
        "dashboard.html",
        org_name=session.get("org_name", "User"),
        cars=ml_model.known_cars(),
        fuels=ml_model.FUEL_OPTIONS,
        types=ml_model.TYPE_OPTIONS,
        bands=ml_model.band_info(),
    )


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    try:
        prediction = ml_model.predict_price(data)
    except ml_model.UnknownCar as exc:
        message = (f"\u201c{exc.name}\u201d is not in the dataset." if exc.name
                   else "Choose a car first.")
        return jsonify(error=message, suggestions=exc.suggestions), 400
    except Exception:
        log.exception("Prediction failed")
        return jsonify(error="The price model could not run. Check the terminal window for details."), 500

    db = get_db()
    db.execute(
        """INSERT INTO predictions (user_id, car_model, fuel, transmission, result, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session["user_id"], prediction["matched_model"], prediction["fuel"],
         prediction["transmission"], prediction["estimated_range"],
         datetime.now().strftime("%d %b, %I:%M %p")),
    )
    db.commit()
    return jsonify(prediction)


@app.route("/api/history")
@login_required
def api_history():
    rows = get_db().execute(
        """SELECT car_model, fuel, transmission, result, created_at FROM predictions
           WHERE user_id = ? ORDER BY id DESC LIMIT 10""",
        (session["user_id"],),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


# --------------------------------------------------------------------------
# Start-up
# --------------------------------------------------------------------------
init_db()
try:
    _state = ml_model.get_state()
    log.info("Price model ready (%s, %d cars)", _state.source, len(_state.cars))
except Exception:
    log.exception("The price model could not be loaded")


def free_port(preferred):
    """First free port at or after `preferred` (5000 is taken by AirPlay on newer Macs)."""
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no traffic is sent
            return s.getsockname()[0]
    except OSError:
        return None


if __name__ == "__main__":
    port = free_port(int(os.environ.get("PORT", 5000)))
    ip = lan_ip()
    print()
    print("  Car price AI is running.")
    print(f"  On this computer:   http://127.0.0.1:{port}")
    if ip:
        print(f"  On your phone/tablet (same Wi-Fi):   http://{ip}:{port}")
    print("  Demo login:         demo@carai.com / demo123")
    print("  Press Ctrl+C to stop.")
    print()
    if not os.environ.get("NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=bool(os.environ.get("CARAI_DEBUG")),
            threaded=True, use_reloader=False)
