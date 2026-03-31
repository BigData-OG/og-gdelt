"""
GDELT Stock Prediction API
 
Cloud Run API that orchestrates ML training and prediction
for stock price prediction using GDELT news sentiment.
"""
 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
 
from train_pipeline import submit_training_job
from services.data_extractor import DataExtractor
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI(title="GDELT Stock Prediction API")
 
# Initialize the data extractor
extractor = DataExtractor()
 
 
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
 
 
# --- Endpoints ---
 
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
 
 
@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Submit a training pipeline for a given company.
 
    This endpoint:
    1. Extracts GDELT sentiment data from BigQuery
    2. Extracts stock price data from yFinance
    3. Joins, cleans, and feature-engineers the data
    4. Submits a Vertex AI training job
 
    The training job runs asynchronously (~2-5 minutes).
 
    Args:
        request: TrainRequest with company_name, ticker, and optional params
 
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
            data_path = f"companies/{ticker}/processed/training_data*.csv"
            logger.info(f"  Skipping extraction, using existing data at: {data_path}")
 
        # The training script expects a path relative to the bucket (no gs:// prefix)
        # Strip the gs://bucket_name/ prefix if present
        bucket_prefix = "gs://og-gdelt-main-data-dev/"
        if data_path.startswith(bucket_prefix):
            relative_path = data_path[len(bucket_prefix):]
        else:
            relative_path = data_path
 
        # Step 2: Submit training job to Vertex AI
        logger.info(f"Step 2: Submitting Vertex AI training job for {ticker}")
        result = submit_training_job(
            ticker=ticker,
            input_data_path=relative_path,
        )
 
        logger.info(f"=== Training pipeline complete for {ticker} ===")
        logger.info(f"  Job: {result['job_name']}")
 
        return TrainResponse(
            **result,
            extraction_summary=extraction_summary,
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
 