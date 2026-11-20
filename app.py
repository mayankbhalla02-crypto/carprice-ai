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
def _hash(password):
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


def _current_user(request):
    return SESSIONS.get(request.cookies.get("session", ""))


def _auth_context(request, **extra):
    try:
        bands = ml_model.band_info()
        count = len(ml_model.known_cars())
    except Exception as exc:
        print("ml_model failed:", repr(exc), flush=True)
        bands, count = [], 0
    ctx = {"request": request, "bands": bands, "car_count": count}
    ctx.update(extra)
    return ctx


def _login_response(email):
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = email
    resp = RedirectResponse(url="/dashboard", status_code=303)
    resp.set_cookie("session", token, httponly=True, samesite="lax")
    return resp


def _password_error(password):
    if len(password) > 5:
        return None
    return "Password must be at least 6 characters."


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
        else:
            error = _password_error(password)
        if not error:
            conn = _db()
            try:
                conn.execute(
                    "INSERT INTO users (email, org_name, password_hash) VALUES (?, ?, ?)",
                    (email, org_name, _hash(password)),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                error = "That email is already registered."
            finally:
                conn.close()
        if not error:
            return _login_response(email)
        return templates.TemplateResponse(
            request=request,
            name="signup.html",
            context=_auth_context(request, error=error),
            status_code=400,
        )
    return templates.TemplateResponse(
        request=request, name="signup.html", context=_auth_context(request)
    )


@app.api_route("/dashboard", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def dashboard(request: Request):
    if not _current_user(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request=request, name="dashboard.html", context={"request": request}
    )


@app.get("/logout")
async def logout(request: Request):
    SESSIONS.pop(request.cookies.get("session", ""), None)
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie("session")
    return resp


# ---------------------------------------------------------------- car API
@app.get("/api/cars")
async def api_cars():
    """Cars that have a real listed price on CarDekho (no placeholders)."""
    return [
        {"model": c["model"], "price": ml_model._display_price(c["price"])}
        for c in ml_model.known_cars()
        if c["price"].startswith("Rs.")
    ]


@app.post("/api/price")
async def api_price(model: str = Form(...)):
    try:
        result = ml_model.predict_price({"model": model})
    except ml_model.UnknownCar as e:
        return JSONResponse(
            status_code=404,
            content={
                "status": "unknown",
                "message": "'" + e.name + "' isn't in the car list.",
                "suggestions": e.suggestions,
            },
        )
    return {"status": "success", **result}


@app.post("/api/estimate")
async def api_estimate(
    model: str = Form(...),
    year: int = Form(...),
    kms_driven: float = Form(...),
    fuel: str = Form("Petrol"),
    transmission: str = Form("Manual"),
):
    try:
        result = ml_model.estimate_used_price(
            {
                "model": model,
                "year": year,
                "kms_driven": kms_driven,
                "fuel": fuel,
                "transmission": transmission,
            }
        )
    except ml_model.UnknownCar as e:
        return JSONResponse(
            status_code=404,
            content={
                "status": "unknown",
                "message": "'" + e.name + "' isn't in the car list.",
                "suggestions": e.suggestions,
            },
        )
    return {"status": "success", **result}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
