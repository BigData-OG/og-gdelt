from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from train import train_local_model

router = APIRouter()

# Load the model at startup
try:
    model = joblib.load('train_results_vertex_model_amazon_model_model.joblib')
except FileNotFoundError:
    model = None


class PredictRequest(BaseModel):
    # not really needed, this version will only do one model
    # company_name: str
    # ticker: str
    daily_exposure_count: float
    daily_avg_tone: float
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float
    daily_return_pct: float
    day_of_week: int


@router.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Make sure 'train_results_vertex_model_amazon_model_model.joblib' is in the correct path.")

    try:
        # Create a DataFrame from the request
        data = {
            "daily_exposure_count": [req.daily_exposure_count],
            "daily_avg_tone": [req.daily_avg_tone],
            "Open": [req.Open],
            "High": [req.High],
            "Low": [req.Low],
            "Close": [req.Close],
            "Volume": [req.Volume],
            "daily_return_pct": [req.daily_return_pct],
            "day_of_week": [req.day_of_week],
        }
        df = pd.DataFrame(data)

        # Ensure the column order matches the model's expectation
        feature_columns = [
            "Open", "High", "Low", "Close", "Volume",
            "daily_return_pct", "day_of_week", "daily_exposure_count", "daily_avg_tone"
        ]
        df = df[feature_columns]

        # Make prediction
        prediction = model.predict(df)

        # Convert numpy types to native Python types for JSON serialization
        if isinstance(prediction, np.ndarray):
            prediction_list = prediction.tolist()
        else:
            prediction_list = [prediction]

        return {"prediction": prediction_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TrainRequest(BaseModel):
    ticker: str


@router.post("/train")
def run_training(req: TrainRequest):
    try:
        # Call the local training function
        result = train_local_model(req.ticker)

        # Reload the global model in memory so the /predict route uses the freshly trained one
        global model
        model = joblib.load('train_results_vertex_model_amazon_model_model.joblib')

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
