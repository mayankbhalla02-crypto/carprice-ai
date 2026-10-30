import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cardekho.csv"
MODEL_PATH = BASE_DIR / "car_model_rf.pkl"

def train_model():
    """Trains a simple model on Cardekho.csv if model doesn't exist yet."""
    if not DATA_PATH.exists():
        return False
    
    df = pd.read_csv(DATA_PATH)
    # Target: max_price
    if "max_price" not in df.columns:
        return False
        
    y = pd.qcut(df["max_price"].fillna(df["max_price"].median()), q=3, labels=[0, 1, 2])
    X = df[['model', 'fuel', 'Type']].fillna("Unknown")
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_encoded, y)
    
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump((model, X_encoded.columns.tolist()), MODEL_PATH)
    return True

def predict_price(payload):
    if not MODEL_PATH.exists():
        success = train_model()
        if not success:
            return {"predicted_category": "Mid-Range", "estimated_lakhs": "10.0 - 15.0 Lakh"}
            
    model, train_columns = joblib.load(MODEL_PATH)
    
    # Prepare input dataframe matching training columns
    input_data = pd.DataFrame([{
        'model': payload.get('model', 'Maruti Suzuki Jimny'),
        'fuel': payload.get('fuel', 'Petrol'),
        'Type': payload.get('Type', 'Manual')
    }])
    
    input_encoded = pd.get_dummies(input_data)
    # Align columns with training feature set
    for col in train_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[train_columns]
    
    pred_class = model.predict(input_encoded)[0]
    
    # Map category to estimated price ranges
    ranges = {0: "5.0 - 9.5 Lakh (Budget)", 1: "10.0 - 18.0 Lakh (Mid-Range)", 2: "20.0+ Lakh (Premium/Luxury)"}
    return {
        "price_category": int(pred_class),
        "estimated_range": ranges.get(pred_class, "10.0 - 15.0 Lakh")
    }