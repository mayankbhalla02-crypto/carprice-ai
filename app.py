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

# Estimated INR price ranges per quantile class (0: Budget, 1: Mid, 2: Premium)
PRICE_BANDS = {
    0: {
        "fair_num": 350000,
        "low_num": 220000,
        "high_num": 480000,
        "tier_label": "Budget Tier (Class 0)",
    },
    1: {
        "fair_num": 650000,
        "low_num": 480000,
        "high_num": 850000,
        "tier_label": "Mid-Market Tier (Class 1)",
    },
    2: {
        "fair_num": 1250000,
        "low_num": 850000,
        "high_num": 1800000,
        "tier_label": "Premium Tier (Class 2)",
    },
}


def format_inr(amount: float) -> str:
    return f"₹ {amount:,.0f}"


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
    car_name: str = Form(""),
):
    try:
        if not EXPECTED_FEATURES:
            raise ValueError("Model does not expose feature_names_in_.")

        row = {col: 0 for col in EXPECTED_FEATURES}

        age = max(0, 2026 - year)
        log_kms = float(np.log1p(max(0.0, kms_driven)))
        fuel_token = fuel.strip().lower()
        trans_token = transmission.strip().lower()
        car_token = car_name.strip().lower()

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
            elif car_token and c_low.startswith("model_") and car_token in c_low:
                row[col] = 1

        X_input = pd.DataFrame([row], columns=EXPECTED_FEATURES)
        pred_raw = model.predict(X_input)[0]

        try:
            pred_class = int(pred_raw)
        except (ValueError, TypeError):
            pred_class = 1

        band = PRICE_BANDS.get(pred_class, PRICE_BANDS[1])

        # Slight age & usage adjustment around the band's center
        age_factor = max(0.75, 1.0 - (age * 0.03))
        km_factor = max(0.80, 1.0 - (kms_driven / 500000.0))
        adj = (age_factor + km_factor) / 2.0

        fair_val = band["fair_num"] * adj
        low_val = band["low_num"] * adj
        high_val = band["high_num"] * adj

        title_prefix = f"{car_name.strip()} • " if car_name.strip() else ""

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "fairPriceDisplay": f"{title_prefix}{format_inr(fair_val)} ({band['tier_label']})",
                "lowEstDisplay": format_inr(low_val),
                "highEstDisplay": format_inr(high_val),
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
