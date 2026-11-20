import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import ml_model                      # <-- import goes up here

app = FastAPI(title="CarPrice AI - Valuation Platform")   # <-- app is created here

# ... static mount, templates, model load, your existing routes ...

@app.get("/api/cars")                # <-- new routes go down here
async def api_cars():
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
                "message": f"'{e.name}' isn't in the car list.",
                "suggestions": e.suggestions,
            },
        )
    return {"status": "success", **result}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
