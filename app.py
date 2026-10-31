from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="FreshRoute AI - Car Price Prediction")

# Set up templates folder for HTML rendering
templates = Jinja2Templates(directory="templates")

# Load your pre-trained Random Forest model
model = joblib.load("car_model_rf.pkl")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main login or home page."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serves your dashboard.html UI."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/predict")
async def predict(
    year: int = Form(...),
    kms_driven: float = Form(...),
    fuel: str = Form(...),
    transmission: str = Form(...)
):
    """Receives vehicle details, computes feature engineering, and returns Random Forest prediction."""
    try:
        # Feature Engineering matching your training pipeline
        current_year = 2026
        car_age = current_year - year
        log_kms = np.log(kms_driven) if kms_driven > 0 else 0

        # Construct input DataFrame / Array matching model features
        # (Adjust columns to match the exact format your Random Forest pipeline expects)
        input_data = pd.DataFrame([[car_age, log_kms, fuel, transmission]], 
                                  columns=['age', 'log_kms', 'fuel', 'transmission'])
        
        # Perform prediction
        predicted_price = model.predict(input_data)[0]
        
        # Price range brackets
        fair_price = round(predicted_price, 2)
        low_price = round(fair_price * 0.92, 2)
        high_price = round(fair_price * 1.08, 2)

        return JSONResponse(status_code=200, content={
            "status": "success",
            "fairPriceDisplay": f"₹ {fair_price:,.2f}",
            "lowEstDisplay": f"₹ {low_price:,.2f}",
            "highEstDisplay": f"₹ {high_price:,.2f}"
        })
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str-e})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
