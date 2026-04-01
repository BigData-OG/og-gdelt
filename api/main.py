from __future__ import annotations

import logging
from io import StringIO
import os
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from google.cloud import aiplatform, storage, firestore
from fastapi import FastAPI, APIRouter, HTTPException

# to pull in train routes
from api.train_routes import router as train_router

# from api.services.data_extractor import DataExtractor
from api.services.data_extractor import DataExtractor
from api.services.model_repository import ModelRepository

logger = logging.getLogger(__name__)
app = FastAPI()
router = APIRouter(prefix="", tags=["predict"])

PROJECT_ID = os.environ.get("PROJECT_ID", "gdelt-stock-sentiment-analysis")
REGION = os.environ.get("GCP_REGION", "us-west1")
BUCKET =    os.environ.get("BUCKET_NAME", "og-gdelt-main-data-dev")
FIRESTORE_DB = os.environ.get("FIRESTORE_DB_NAME", "og-gdelt-dev-firestore-db")
# endpoints
firestore_client = firestore.Client(project=PROJECT_ID,database=FIRESTORE_DB)
model_repository = ModelRepository(firestore_client)

COMPANY_TO_TICKER = {
    "amazon": "AMZN",
    "pfizer": "PFE",
    "aramco": "2222.SR",
}

FEATURE_COLUMNS = [
    "daily_exposure_count",
    "daily_avg_tone",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "daily_return_pct",
    "day_of_week",
]


class PredictRequest(BaseModel):
    company_name: str = Field(..., description="Company name, e.g. Amazon")
    ticker: str = Field(..., description="Stock ticker, e.g. AMZN")


class PredictResponse(BaseModel):
    company_name: str
    ticker: Optional[str]
    status: str
    prediction: Optional[Any] = None
    processed_path: Optional[str] = None
    latest_event_date: Optional[str] = None
    features_used: Optional[list] = None
    message: Optional[str] = None


def normalize_company_name(name: str) -> str:
    return name.strip().lower()


def resolve_ticker(company_name: str) -> Optional[str]:
    return COMPANY_TO_TICKER.get(normalize_company_name(company_name))



def read_gcs_csv(gcs_path: str) -> pd.DataFrame:
    """
    read GCS  CSV

    """
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    path_without_prefix = gcs_path.replace("gs://", "", 1)
    bucket_name, blob_path = path_without_prefix.split("/", 1)

    storage_client = storage.Client(project=PROJECT_ID)

    if "*" in blob_path:
        prefix = blob_path.split("*", 1)[0]
        blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix))
        if not blobs:
            raise FileNotFoundError(f"No blobs found for wildcard path: {gcs_path}")

        # 取排序后的最后一个
        blobs = sorted(blobs, key=lambda b: b.name)
        blob = blobs[-1]
    else:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

    content = blob.download_as_text()
    return pd.read_csv(StringIO(content))


def build_latest_instance_from_processed(training_path: str) -> tuple[Dict[str, Any], str]:
    """
    read processed CSV data, order by event_date 
    
    """
    df = read_gcs_csv(training_path)

    if df.empty:
        raise ValueError("Processed CSV is empty.")

    required_cols = ["event_date"] + FEATURE_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in processed CSV: {missing}")

    # only check inference 
    df = df.dropna(subset=required_cols)
    if df.empty:
        raise ValueError("No valid rows remain after dropping null inference features.")

    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date")

    latest = df.iloc[-1]

    # feature and no next_day_close 
    instance = {
        "daily_exposure_count": float(latest["daily_exposure_count"]),
        "daily_avg_tone": float(latest["daily_avg_tone"]),
        "Open": float(latest["Open"]),
        "High": float(latest["High"]),
        "Low": float(latest["Low"]),
        "Close": float(latest["Close"]),
        "Volume": float(latest["Volume"]),
        "daily_return_pct": float(latest["daily_return_pct"]),
        "day_of_week": int(latest["day_of_week"]),
    }

    latest_date = latest["event_date"].strftime("%Y-%m-%d")
    return instance, latest_date


def call_vertex_endpoint(endpoint_id: str, instance: Dict[str, Any]) -> Any:
    aiplatform.init(project=PROJECT_ID, location=REGION)

    endpoint = aiplatform.Endpoint(
        endpoint_name=endpoint_id
    )

    response = endpoint.predict(instances=[instance])

    if not response.predictions:
        raise RuntimeError("Vertex AI returned no predictions.")

    return response.predictions[0]


@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/models/{company_name}")
def model_status(company_name: str) -> Dict[str, Any]:
    ticker = resolve_ticker(company_name)
    model_id = model_repository.get_model_id_by_ticker(company_name) if ticker else None
    if not ticker:
        return {
            "company_name": company_name,
            "ticker": None,
            "status": "unknown_company",
            "has_model": False,
        }

    return {
        "company_name": company_name,
        "ticker": ticker,
        "status": "ready" if model_id else "training_needed",
        "has_model": bool(model_id),
    }


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    company_name = req.company_name.strip()
    ticker = req.ticker.strip()

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported company: {company_name}. "
                   f"Currently supported: {list(COMPANY_TO_TICKER.keys())}"
        )

    # if no model / endpoint, return training_needed
    endpoint_info = model_repository.get_model_id_by_ticker(ticker)
    endpoint_id = endpoint_info if endpoint_info else None

    if not endpoint_id or "REPLACE" in endpoint_id:
        return PredictResponse(
            company_name=company_name,
            ticker=ticker,
            status="training_needed",
            message="No deployed Vertex endpoint exists for this ticker yet."
        )

    # get one latest feature row for inference
    try:
        extractor = DataExtractor(BUCKET, PROJECT_ID)
        latest_data = extractor.get_latest_features(
            company_name=company_name,
            ticker=ticker
        )
        latest_features = latest_data.get("features", {})
    except Exception as e:
        logger.exception("Latest feature extraction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Latest feature extraction failed: {str(e)}"
        )

    required_cols = [
        "daily_exposure_count",
        "daily_avg_tone",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "daily_return_pct",
        "day_of_week",
    ]

    missing = [c for c in required_cols if c not in latest_features]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required inference features: {missing}"
        )

    try:
        instance = {
            "daily_exposure_count": float(latest_features["daily_exposure_count"]),
            "daily_avg_tone": float(latest_features["daily_avg_tone"]),
            "Open": float(latest_features["Open"]),
            "High": float(latest_features["High"]),
            "Low": float(latest_features["Low"]),
            "Close": float(latest_features["Close"]),
            "Volume": float(latest_features["Volume"]),
            "daily_return_pct": float(latest_features["daily_return_pct"]),
            "day_of_week": int(latest_features["day_of_week"]),
        }
        instance = [instance[col] for col in FEATURE_COLUMNS]
        latest_date = str(latest_data.get("event_date"))
    except Exception as e:
        logger.exception("Failed to build inference instance")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build inference input: {str(e)}"
        )

    try:
        prediction = call_vertex_endpoint(endpoint_id, instance)
    except Exception as e:
        logger.exception("Vertex inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Vertex inference failed: {str(e)}"
        )

    return PredictResponse(
        company_name=company_name,
        ticker=ticker,
        status="predicted",
        prediction=prediction,
        latest_event_date=latest_date,
        features_used=instance,
    )
app.include_router(router)
app.include_router(train_router)
