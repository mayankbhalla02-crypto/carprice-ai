import hashlib
import os
import secrets
import sqlite3
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import ml_model

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "carusers.db"

app = FastAPI(title="CarPrice AI - Valuation Platform")

if os.path.isdir(BASE_DIR / "static"):
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

SESSIONS = {}  # token -> email (in memory; users sign in again after a restart)


# ---------------------------------------------------------------- users
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "email TEXT PRIMARY KEY, org_name TEXT, password_hash TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO users (email, org_name, password_hash) VALUES (?, ?, ?)",
        ("demo@carai.com", "Demo", _hash("demo123")),
    )
    conn.commit()
    return conn


def _current_user(request: Request):
    return SESSIONS.get(request.cookies.get("session", ""))


def _auth_context(request, **extra):
    try:
        bands = ml_model.band_info()
        count = len(ml_model.known_cars())
    except Exception:
        bands, count = [], 0
    ctx = {"request": request, "bands": bands, "car_count": count}
    ctx.update(extra)
    return ctx


def _login_response(email: str):
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = email
    resp = RedirectResponse(url="/dashboard", status_code=303)
    resp.set_cookie("session", token, httponly=True, samesite="lax")
    return resp


# ---------------------------------------------------------------- pages
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="login.html", context=_auth_context(request)
    )


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    email = username.strip().lower()
    conn = _db()
    row = conn.execute("SELECT password_hash FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    if row and row[0] == _hash(password):
        return _login_response(email)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context=_auth_context(request, error="Wrong email or password."),
        status_code=401,
    )


@app.api_route("/signup", methods=["GET", "POST"], response_class=HTMLResponse)
async def signup(request: Request):
    if request.method == "POST":
        form = await request.form()
        email = str(form.get("email", "")).strip().lower()
        org_name = str(form.get("org_name", "")).strip()
        password = str(form.get("password", ""))
        error = None
        if "@" not in email:
            error = "Enter a valid email."
        elif len(password)
