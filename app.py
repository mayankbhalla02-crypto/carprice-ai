import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd

app = FastAPI(title="FreshRoute AI - Car Price Prediction")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model = joblib.load("car_model_rf.pkl")
EXPECTED_FEATURES = list(getattr(model, "feature_names_in_", []))

# Price band display mapping for classes 0, 1, 2 (from qcut q=3)
BAND_DISPLAY = {
    0: {
        "fair": "Budget Band (Tier 1 / 3)",
        "low": "Lower Market Tier",
        "high": "Entry-to-Mid Tier",
    },
    1: {
        "fair": "Mid-Market Band (Tier 2 / 3)",
        "low": "Mid-Low Tier",
        "high": "Mid-High Tier",
    },
    2: {
        "fair": "Premium Band (Tier 3 / 3)",
        "low": "Upper-Mid Tier",
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

        # Start with all expected dummy/feature columns set to 0
        row = {col: 0 for col in EXPECTED_FEATURES}

        fuel_clean = fuel.strip().lower()
        trans_clean = transmission.strip().lower()

        # Set matching fuel dummy column if present (e.g., Fuel_Petrol/CNG, Fuel_Diesel, etc.)
        for col in EXPECTED_FEATURES:
            col_low = col.lower()
            if col_low.startswith("fuel_"):
                if fuel_clean in col_low:
                    row[col] = 1
            elif col_low.startswith("type_") or col_low.startswith("transmission_"):
                if trans_clean in col_low:
                    row[col] = 1

        input_data = pd.DataFrame([row], columns=EXPECTED_FEATURES)
        pred_class = int(model.predict(input_data)[0])
        info = BAND_DISPLAY.get(pred_class, BAND_DISPLAY[1])

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "fairPriceDisplay": info["fair"],
                "lowEstDisplay": info["low"],
                "highEstDisplay": info["high"],
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
