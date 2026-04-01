"""
Training routes - /train and /train/status endpoints.
 
Uses APIRouter so it can be plugged into the main FastAPI app in main.py.
"""
 
import os
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from google.cloud import storage as gcs_storage, firestore

import logging
 
from api.services.model_repository import ModelRepository
from api.train_pipeline import (
    submit_training_job,
    train_and_deploy_background,
    job_tracker,
)
from api.services import DataExtractor
 
logger = logging.getLogger(__name__)
 
router = APIRouter(prefix="", tags=["train"])
PROJECT_ID = os.environ.get("PROJECT_ID", "gdelt-stock-sentiment-analysis")
REGION = os.environ.get("GCP_REGION", "us-west1")
BUCKET_NAME =    os.environ.get("BUCKET_NAME", "og-gdelt-main-data-dev")
# Initialize the data extractor and GCS client
extractor = DataExtractor(bucket=BUCKET_NAME, project_id=PROJECT_ID)
storage_client = gcs_storage.Client(project=PROJECT_ID)

 
 
# --- Request/Response models ---
 
class TrainRequest(BaseModel):
    """Request body for /train endpoint."""
    company_name: str
    ticker: str
    years: Optional[int] = 5
    skip_extraction: Optional[bool] = False
 
 
class TrainResponse(BaseModel):
    """Response from /train endpoint."""
    status: str
    job_name: str
    display_name: str
    ticker: str
    data_path: str
    output_dir: str
    extraction_summary: Optional[dict] = None
    deployment: Optional[str] = None

class ModelAlreadyExistResponse(BaseModel):
    status: str
    message: str
 
 
# --- Helper functions ---
 
def resolve_gcs_wildcard(path_with_wildcard: str) -> str:
    """
    Resolve a GCS wildcard path to an actual filename.
    BigQuery exports use wildcards (e.g., training_data*.csv) which get
    replaced with shard numbers (e.g., training_data000000000000.csv).
    """
    if "*" not in path_with_wildcard:
        return path_with_wildcard
 
    prefix = path_with_wildcard.split("*")[0]
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix))
    csv_files = [blob.name for blob in blobs if blob.name.endswith(".csv")]
 
    if not csv_files:
        raise ValueError(
            f"No data files found matching '{path_with_wildcard}' in bucket '{BUCKET_NAME}'. "
            f"Data extraction may have failed."
        )
 
    logger.info(f"  Resolved wildcard: found {len(csv_files)} file(s), using: {csv_files[0]}")
    return csv_files[0]
 
 
# --- Endpoints ---
 
@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Submit a training pipeline for a given company.
 
    This endpoint:
    1. Extracts GDELT sentiment data from BigQuery
    2. Extracts stock price data from yFinance
    3. Joins, cleans, and feature-engineers the data
    4. Submits a Vertex AI training job
    5. Kicks off background task to auto-deploy the model
 
    Check /train/status/{ticker} for progress.
    """
    ticker = request.ticker.upper()
    company_name = request.company_name
    firestore_client = firestore.Client(project=PROJECT_ID,database=os.environ.get("FIRESTORE_DB_NAME", "og-gdelt-dev-firestore-db"))
    repo = ModelRepository(firestore_client)
    model_endpoint = repo.get_model_id_by_company_name(company_name)
    if model_endpoint is not None:
        return ModelAlreadyExistResponse(
            status="model_exists",
            message=f"A model for '{company_name}' already exists at endpoint '{model_endpoint}'. "
                    f"Please delete the existing model before training a new one."
        )
    logger.info(f"=== Training pipeline started for {company_name} ({ticker}) ===")
 
    extraction_summary = None
 
    try:
        # Step 1: Extract and prepare data (unless skipped)
        if not request.skip_extraction:
            logger.info(f"Step 1: Extracting data for {company_name} ({ticker})")
            extraction_result = extractor.extract_company_data(
                company_name=company_name,
                ticker=ticker,
                years=request.years,
            )
            data_path = extraction_result["paths"]["processed"]
            extraction_summary = extraction_result.get("row_counts")
            logger.info(f"  Data extracted to: {data_path}")
        else:
            data_path = f"gs://{BUCKET_NAME}/companies/{ticker}/processed/training_data*.csv"
            logger.info(f"  Skipping extraction, using existing data at: {data_path}")
 
        # Strip the gs://bucket_name/ prefix to get relative path
        bucket_prefix = f"gs://{BUCKET_NAME}/"
        if data_path.startswith(bucket_prefix):
            relative_path = data_path[len(bucket_prefix):]
        else:
            relative_path = data_path
 
        # Resolve wildcard to actual filename
        relative_path = resolve_gcs_wildcard(relative_path)
 
        # Step 2: Submit training job to Vertex AI
        logger.info(f"Step 2: Submitting Vertex AI training job for {ticker}")
        result = submit_training_job(
            ticker=ticker,
            input_data_path=relative_path,
        )
 
        # Step 3: Start background task to wait for training, then deploy
        background_tasks.add_task(
            train_and_deploy_background,
            job_name=result["job_name"],
            ticker=ticker,
        )
        deployment_msg = (
            f"Background deployment started. "
            f"Check /train/status/{ticker.lower().replace('.', '-').replace('/', '-')} for progress."
        )
 
        logger.info(f"=== Training pipeline complete for {ticker} ===")
        logger.info(f"  Job: {result['job_name']}")
        logger.info(f"  Background deployment task started")
 
        return TrainResponse(
            **result,
            extraction_summary=extraction_summary,
            deployment=deployment_msg,
        )
 
    except ValueError as e:
        logger.error(f"Data error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
 
    except Exception as e:
        logger.error(f"Error in training pipeline for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Training pipeline failed: {str(e)}",
        )
 
 
@router.get("/train/status/{ticker}")
async def get_train_status(ticker: str):
    """
    Check the status of a training + deployment pipeline.
 
    Possible statuses:
    - training, registering_model, creating_endpoint, deploying, deployed
    - failed, error, not_found
    """
    safe_name = ticker.lower().replace(".", "-").replace("/", "-")
 
    if safe_name in job_tracker:
        return job_tracker[safe_name]
 
    return {
        "status": "not_found",
        "message": f"No training job found for ticker '{ticker}'. Submit a /train request first.",
    }