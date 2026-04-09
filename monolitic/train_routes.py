import os
import pandas as pd
import joblib
import numpy as np

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from google.cloud import storage as gcs_storage
from data_extractor import DataExtractor
from train_pipeline import train_and_deploy_background

router = APIRouter()

# --- GCP Clients & Helpers ---
PROJECT_ID = os.environ.get("PROJECT_ID", "gdelt-stock-sentiment-analysis")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "og-gdelt-main-data-dev")

extractor = DataExtractor(BUCKET_NAME, PROJECT_ID)
storage_client = gcs_storage.Client()


def resolve_gcs_wildcard(path_with_wildcard: str) -> str:
    """Resolves BigQuery wildcards to exact GCS filenames."""
    if "*" not in path_with_wildcard:
        return path_with_wildcard

    prefix = path_with_wildcard.split("*")[0]
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix))
    csv_files = [blob.name for blob in blobs if blob.name.endswith(".csv")]

    if not csv_files:
        raise ValueError(f"No data files found matching '{path_with_wildcard}'")
    return csv_files[0]


# --- Dynamic Model Cache ---
loaded_models = {}


def get_model(ticker: str):
    """Lazy loader: Checks memory first, then disk, otherwise fails."""
    ticker = ticker.upper()

    if ticker in loaded_models:
        return loaded_models[ticker]

    file_path = f"{ticker}_model.joblib"
    if os.path.exists(file_path):
        print(f"Loading {ticker} model from disk into memory...")
        loaded_models[ticker] = joblib.load(file_path)
        return loaded_models[ticker]

    return None


# --- API Routes ---

class TrainRequest(BaseModel):
    company_name: str
    ticker: str
    years: int = 5


@router.post("/train")
def run_training(req: TrainRequest, background_tasks: BackgroundTasks):
    try:
        # 1. Extract data using BigQuery
        print(f"Extracting data for {req.company_name} ({req.ticker})")
        extraction_result = extractor.extract_company_data(
            company_name=req.company_name,
            ticker=req.ticker,
            years=req.years
        )

        # 2. Resolve the GCS path
        data_path = extraction_result["paths"]["processed"]
        bucket_prefix = f"gs://{BUCKET_NAME}/"
        if data_path.startswith(bucket_prefix):
            relative_path = data_path[len(bucket_prefix):]
        else:
            relative_path = data_path

        exact_csv_path = resolve_gcs_wildcard(relative_path)
        full_gcs_uri = f"gs://{BUCKET_NAME}/{exact_csv_path}"

        # 3. Trigger the LOCAL training in the background
        background_tasks.add_task(
            train_and_deploy_background,
            ticker=req.ticker,
            data_path=full_gcs_uri
        )

        return {
            "status": "extraction_complete_training_started",
            "message": f"Data extracted to {full_gcs_uri}. Local container training started in background.",
            "ticker": req.ticker
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictRequest(BaseModel):
    ticker: str
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
    model = get_model(req.ticker)

    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model for {req.ticker} not found. Please run /train first."
        )

    try:
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

        feature_columns = [
            "Open", "High", "Low", "Close", "Volume",
            "daily_return_pct", "day_of_week", "daily_exposure_count", "daily_avg_tone"
        ]
        df = df[feature_columns]

        prediction = model.predict(df)

        if isinstance(prediction, np.ndarray):
            prediction_list = prediction.tolist()
        else:
            prediction_list = [prediction]

        return {"ticker": req.ticker.upper(), "prediction": prediction_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
