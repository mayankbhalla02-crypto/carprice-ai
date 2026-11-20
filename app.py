from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

app = FastAPI(title="FreshRoute AI - Car Price Prediction")
templates = Jinja2Templates(directory="templates")
model = joblib.load("car_model_rf.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"request": request},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.post("/predict")
async def predict(
    year: int = Form(...),
    kms_driven: float = Form(...),
    fuel: str = Form(...),
    transmission: str = Form(...),
):
    try:
        current_year = 2026
        car_age = max(0, current_year - year)
        log_kms = float(np.log1p(max(0.0, kms_driven)))

        # Normalize fuel string (e.g., "Petrol", "Diesel", "CNG", "Electric")
        fuel_clean = fuel.strip().title()
        transmission_clean = transmission.strip().title()

        input_data = pd.DataFrame(
            [[car_age, log_kms, fuel_clean, transmission_clean]],
            columns=["age", "log_kms", "fuel", "transmission"],
        )

        predicted_price = float(model.predict(input_data)[0])

        fair_price = round(predicted_price, 2)
        low_price = round(fair_price * 0.92, 2)
        high_price = round(fair_price * 1.08, 2)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "fairPriceDisplay": f"₹ {fair_price:,.2f}",
                "lowEstDisplay": f"₹ {low_price:,.2f}",
                "highEstDisplay": f"₹ {high_price:,.2f}",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
