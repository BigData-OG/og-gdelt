"""
GDELT Stock Prediction API
 
Cloud Run API that orchestrates ML training and prediction
for stock price prediction using GDELT news sentiment.
"""
 
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from google.cloud import storage as gcs_storage
import logging
 
from train_pipeline import (
    submit_training_job,
    train_and_deploy_background,
    job_tracker,
)
from services.data_extractor import DataExtractor
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI(title="GDELT Stock Prediction API")
 
# Initialize the data extractor and GCS client
extractor = DataExtractor()
storage_client = gcs_storage.Client()
BUCKET_NAME = "og-gdelt-main-data-dev"
 
 
# --- Request/Response models ---
 
class TrainRequest(BaseModel):
    """Request body for /train endpoint."""
    company_name: str  # e.g., "Tesla", "Google", "Apple"
    ticker: str        # e.g., "TSLA", "GOOGL", "AAPL"
    years: Optional[int] = 5  # years of historical data (default 5)
    skip_extraction: Optional[bool] = False  # skip data extraction if data already exists
 
 
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
 
 
# --- Helper functions ---
 
def resolve_gcs_wildcard(path_with_wildcard: str) -> str:
    """
    Resolve a GCS wildcard path to an actual filename.
 
    BigQuery exports use wildcards (e.g., training_data*.csv) which get
    replaced with shard numbers (e.g., training_data000000000000.csv).
    The training script needs the exact filename, not the wildcard.
 
    This function lists files in GCS matching the prefix and returns
    the first matching CSV file.
 
    Args:
        path_with_wildcard: Path relative to bucket, e.g.,
                            "companies/TSLA/processed/training_data*.csv"
 
    Returns:
        Exact path like "companies/TSLA/processed/training_data000000000000.csv"
    """
    if "*" not in path_with_wildcard:
        return path_with_wildcard
 
    # Strip the wildcard to get the prefix
    # "companies/TSLA/processed/training_data*.csv" -> "companies/TSLA/processed/training_data"
    prefix = path_with_wildcard.split("*")[0]
 
    # List all files in GCS that start with this prefix
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix))
 
    # Filter to just CSV files
    csv_files = [blob.name for blob in blobs if blob.name.endswith(".csv")]
 
    if not csv_files:
        raise ValueError(
            f"No data files found matching '{path_with_wildcard}' in bucket '{BUCKET_NAME}'. "
            f"Data extraction may have failed."
        )
 
    logger.info(f"  Resolved wildcard: found {len(csv_files)} file(s), using: {csv_files[0]}")
    return csv_files[0]
 
 
# --- Endpoints ---
 
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
 
 
@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Submit a training pipeline for a given company.
 
    This endpoint:
    1. Extracts GDELT sentiment data from BigQuery
    2. Extracts stock price data from yFinance
    3. Joins, cleans, and feature-engineers the data
    4. Submits a Vertex AI training job
    5. Kicks off a background task that waits for training to finish,
       then automatically registers and deploys the model
 
    The training job runs asynchronously (~2-5 minutes).
    Model deployment runs in the background (~15-20 minutes).
    Check /train/status/{ticker} for progress.
 
    Args:
        request: TrainRequest with company_name, ticker, and optional params
        background_tasks: FastAPI BackgroundTasks (injected automatically)
 
    Returns:
        Job details including job_name for tracking
    """
    ticker = request.ticker.upper()
    company_name = request.company_name
 
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
            # Use existing processed data
            data_path = f"gs://{BUCKET_NAME}/companies/{ticker}/processed/training_data*.csv"
            logger.info(f"  Skipping extraction, using existing data at: {data_path}")
 
        # Strip the gs://bucket_name/ prefix to get relative path
        bucket_prefix = f"gs://{BUCKET_NAME}/"
        if data_path.startswith(bucket_prefix):
            relative_path = data_path[len(bucket_prefix):]
        else:
            relative_path = data_path
 
        # Resolve wildcard to actual filename
        # BigQuery exports use wildcards but pd.read_csv() needs exact filenames
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
        # Data-related errors (no stock data found, etc.)
        logger.error(f"Data error for {ticker}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
 
    except Exception as e:
        logger.error(f"Error in training pipeline for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Training pipeline failed: {str(e)}",
        )
 
 
@app.get("/train/status/{ticker}")
async def get_train_status(ticker: str):
    """
    Check the status of a training + deployment pipeline.
 
    After calling /train, use this endpoint to track progress.
    
    Possible statuses:
    - training: Vertex AI is training the model
    - registering_model: Training done, uploading to Model Registry
    - creating_endpoint: Setting up the prediction endpoint
    - deploying: Deploying model to endpoint (~15-20 min)
    - deployed: Done! Ready for predictions via /predict
    - failed: Training job failed
    - error: Unexpected error in the pipeline
    - not_found: No training job exists for this ticker
 
    Args:
        ticker: Stock ticker symbol (e.g., "AMZN", "tsla", "2222-SR")
    """
    safe_name = ticker.lower().replace(".", "-").replace("/", "-")
 
    if safe_name in job_tracker:
        return job_tracker[safe_name]
 
    return {
        "status": "not_found",
        "message": f"No training job found for ticker '{ticker}'. Submit a /train request first.",
    }
 