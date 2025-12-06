from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Backend.schemas.input_models import CropYieldInput, ForecastInput
import joblib
import pandas as pd
import os

app = FastAPI(title="Crop Yield Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once when app starts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CATBOOST_MODEL_PATH = os.path.join(BASE_DIR, "models", "CatBoost.pkl")
PROPHET_MODEL_PATH = os.path.join(BASE_DIR, "models", "Prophet.pkl")

model = joblib.load(CATBOOST_MODEL_PATH)
prophet_model = joblib.load(PROPHET_MODEL_PATH)

@app.post("/predict_yield")
def predict_yield(data: CropYieldInput):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    return {"Predicted_Crop_Yield": round(prediction, 2)}


@app.post("/forecast")
def forecast_crop_yield(data: ForecastInput):
    future_dates = pd.date_range(start=data.start_date, periods=data.periods, freq="D")
    future = pd.DataFrame({"ds": future_dates})

    future["Soil_pH"] = data.Soil_pH
    future["Temperature"] = data.Temperature
    future["Humidity"] = data.Humidity
    future["Wind_Speed"] = data.Wind_Speed
    future["N"] = data.N
    future["P"] = data.P
    future["K"] = data.K
    future["Soil_Quality"] = data.Soil_Quality
    forecast = prophet_model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
    return {"forecast": result}
