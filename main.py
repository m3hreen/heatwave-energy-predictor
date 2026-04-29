from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

print("Starting API...")

app = FastAPI(
    title="CAISO Electricity Demand Predictor",
    description="Predicts hourly electricity demand (MWh) and grid risk level for the California grid using weather inputs.",
    version="1.0.0"
)
 
try:
    model = joblib.load("model.pkl")
    meta  = joblib.load("model_meta.pkl")
    P70      = meta["p70"]
    P90      = meta["p90"]
    FEATURES = meta["features"]
except FileNotFoundError as e:
    raise RuntimeError(str(e))
 
print("Model loaded successfully")

# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    temperature: float = Field(..., description="Air temperature at 2 m (°C)")
    humidity: float = Field(..., description="Relative humidity at 2 m (%)")
    apparent_temperature: float = Field(..., description="Feels-like temperature (°C)")
    datetime: str = Field(
        ...,
        description="ISO-8601 datetime string, e.g. '2024-08-15T14:00:00'",
        examples=["2024-08-15T14:00:00"]
    )
 
    model_config = {"json_schema_extra": {
        "example": {
            "temperature": 38.5,
            "humidity": 25,
            "apparent_temperature": 40.1,
            "datetime": "2024-08-15T14:00:00"
        }
    }}
 
 
class PredictResponse(BaseModel):
    predicted_demand_mwh: float
    risk_level: str          # "Low" | "Medium" | "High"
    heatwave_flag: bool
    is_weekend: bool
    hour: int
    month: int
    p70_threshold_mwh: float
    p90_threshold_mwh: float
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def label_risk(demand: float) -> str:
    if demand >= P90:
        return "High"
    elif demand >= P70:
        return "Medium"
    return "Low"
 
 
def build_feature_row(req: PredictRequest) -> pd.DataFrame:
    dt = pd.to_datetime(req.datetime)
    row = {
        "temperature": req.temperature,
        "humidity": req.humidity,
        "apparent_temperature": req.apparent_temperature,
        "hour": dt.hour,
        "month": dt.month,
        "heatwave_flag": int(req.temperature >= 32),
        "is_weekend": int(dt.dayofweek >= 5),
    }
    return pd.DataFrame([row])[FEATURES].astype(float)
 
 
# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "CAISO Demand Predictor is running. POST to /predict to get a forecast."}
 
 
@app.get("/health")
def health_check():
    return {"status": "healthy"}
 
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict hourly electricity demand (MWh) and grid risk level.
 
    - **temperature**: Actual air temp in °C
    - **humidity**: Relative humidity 0–100
    - **apparent_temperature**: Feels-like temp in °C
    - **datetime**: The hour you want to forecast (ISO-8601)
    """
    try:
        X = build_feature_row(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse datetime: {e}")
 
    demand = float(model.predict(X)[0])
    dt = pd.to_datetime(req.datetime)
 
    return PredictResponse(
        predicted_demand_mwh=round(demand, 1),
        risk_level=label_risk(demand),
        heatwave_flag=bool(req.temperature >= 32),
        is_weekend=bool(dt.dayofweek >= 5),
        hour=dt.hour,
        month=dt.month,
        p70_threshold_mwh=round(P70, 1),
        p90_threshold_mwh=round(P90, 1),
    )
 
 
@app.get("/thresholds")
def get_thresholds():
    """Return the risk-level thresholds derived from historical training data."""
    return {
        "low_below_mwh": round(P70, 1),
        "medium_p70_to_p90_mwh": {"min": round(P70, 1), "max": round(P90, 1)},
        "high_above_mwh": round(P90, 1),
    }
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)