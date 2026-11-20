import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="CarPrice AI - Valuation Platform")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model = joblib.load("car_model_rf.pkl")
EXPECTED_FEATURES = list(getattr(model, "feature_names_in_", []))

TIER_MAP = {
    0: {
        "fair": "Budget Band (Tier 1 of 3)",
        "low": "Entry / Lower Quantile",
        "high": "Budget Ceiling",
    },
    1: {
        "fair": "Mid-Market Band (Tier 2 of 3)",
        "low": "Mid-Range Floor",
        "high": "Mid-Range Ceiling",
    },
    2: {
        "fair": "Premium Band (Tier 3 of 3)",
        "low": "Upper-Mid Floor",
        "high": "Top Market Tier",
    },
}


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request},
    )


@app.post("/login")
async def login_submit():
    return RedirectResponse(url="/dashboard", status_code=303)


@app.api_route("/signup", methods=["GET", "POST"], response_class=HTMLResponse)
async def signup(request: Request):
    if request.method == "POST":
        return RedirectResponse(url="/dashboard", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="signup.html",
        context={"request": request},
    )


@app.api_route("/dashboard", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"request": request},
    )


@app.post("/predict")
async def predict(
    year: int = Form(...),
    kms_driven: float = Form(...),
    fuel: str = Form(...),
    transmission: str = Form(...),
):
    try:
        if not EXPECTED_FEATURES:
            raise ValueError("Model does not expose feature_names_in_.")

        row = {col: 0 for col in EXPECTED_FEATURES}

        age = max(0, 2026 - year)
        log_kms = float(np.log1p(max(0.0, kms_driven)))
        fuel_token = fuel.strip().lower()
        trans_token = transmission.strip().lower()

        for col in EXPECTED_FEATURES:
            c_low = col.lower()
            if c_low in ("age", "car_age"):
                row[col] = age
            elif c_low in ("year", "model_year"):
                row[col] = year
            elif c_low in ("kms_driven", "km_driven", "mileage"):
                row[col] = kms_driven
            elif c_low in ("log_kms", "log_km"):
                row[col] = log_kms
            elif c_low.startswith("fuel_") and fuel_token in c_low:
                row[col] = 1
            elif (c_low.startswith("type_") or c_low.startswith("transmission_")) and trans_token in c_low:
                row[col] = 1

        X_input = pd.DataFrame([row], columns=EXPECTED_FEATURES)
        pred_raw = model.predict(X_input)[0]

        try:
            pred_key = int(pred_raw)
        except (ValueError, TypeError):
            pred_key = 1

        tier = TIER_MAP.get(pred_key, TIER_MAP[1])

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "fairPriceDisplay": tier["fair"],
                "lowEstDisplay": tier["low"],
                "highEstDisplay": tier["high"],
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
