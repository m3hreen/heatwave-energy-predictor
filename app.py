import joblib
import pandas as pd

#Load trained model and thresholds 
model = joblib.load("model.pkl")
meta  = joblib.load("model_meta.pkl")
 
p70      = meta["p70"]
p90      = meta["p90"]
features = meta["features"]

#Function for Risk label
def label_risk(demand):
    if demand >= p90:
        return "High"
    elif demand >= p70:
        return "Medium"
    return "Low"


#Function for generating an explanation
def build_explanation(temperature, apparent_temperature, hour, is_weekend, risk_level):
    reasons = []
 
    if apparent_temperature >= 35:
        reasons.append("apparent temperature is very high, driving heavy air-conditioning load")
    elif apparent_temperature >= 28:
        reasons.append("apparent temperature is elevated, increasing cooling demand")
 
    if 16 <= hour <= 21:
        reasons.append("the time falls in the evening peak demand window (4 PM – 9 PM)")
    elif 11 <= hour <= 15:
        reasons.append("midday heat is at its peak")
 
    if temperature >= 32:
        reasons.append("conditions meet the heatwave threshold (≥ 32 °C)")
 
    if is_weekend:
        reasons.append("residential usage is higher on weekends")
 
    if not reasons:
        reasons.append("weather and time conditions are within normal ranges")
 
    return f"Demand risk is {risk_level}. Key drivers: " + "; ".join(reasons) + "."

#Zerve app interface (CONFUSING AS HELL BRUH)
def predict(
    temperature: float,
    humidity: float,
    apparent_temperature: float,
    hour: int,
    month: int,
    is_weekend: bool,
) -> dict:
    heatwave_flag = int(temperature >= 32)
    input_row = pd.DataFrame([{
        "temperature":          temperature,
        "humidity":             humidity,
        "apparent_temperature": apparent_temperature,
        "hour":                 hour,
        "month":                month,
        "heatwave_flag":        heatwave_flag,
        "is_weekend":           int(is_weekend),
    }])[features]
    predicted_demand = float(model.predict(input_row)[0])
    risk_level       = label_risk(predicted_demand)
    explanation      = build_explanation(
        temperature, apparent_temperature, hour, is_weekend, risk_level
    )
 
    return {
        "predicted_demand_mwh": round(predicted_demand, 0),
        "risk_level":           risk_level,
        "explanation":          explanation,
    }
