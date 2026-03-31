"""
GDELT Stock Prediction API
 
Cloud Run API that orchestrates ML training and prediction
for stock price prediction using GDELT news sentiment.
"""
 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
 
from train_pipeline import submit_training_job, TICKER_DISPLAY_NAMES
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI(title="GDELT Stock Prediction API")
 
 
# --- Request/Response models ---
 
class TrainRequest(BaseModel):
    """Optional request body for /train endpoint."""
    input_data_path: Optional[str] = None  # Override data path if needed
 
 
class TrainResponse(BaseModel):
    """Response from /train endpoint."""
    status: str
    job_name: str
    display_name: str
    ticker: str
    data_path: str
    output_dir: str
 
 
# --- Endpoints ---
 
@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
 
 
@app.post("/train/{ticker}", response_model=TrainResponse)
async def train_model(ticker: str, request: Optional[TrainRequest] = None):
    """
    Submit a Vertex AI training job for a given stock ticker.
 
    The endpoint kicks off training asynchronously and returns immediately
    with the job ID. Training typically takes ~5 minutes.
 
    Args:
        ticker: Stock ticker symbol (e.g., AMZN, PFE, 2222.SR)
        request: Optional body with custom input_data_path
 
    Returns:
        Job details including job_name for tracking
    """
    ticker = ticker.upper()
 
    # Validate ticker (optional - remove this if you want to allow any ticker)
    known_tickers = list(TICKER_DISPLAY_NAMES.keys())
    if ticker not in known_tickers:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ticker '{ticker}'. Known tickers: {known_tickers}. "
                   f"Once dynamic extraction is ready, any ticker will work."
        )
 
    # Use custom data path if provided, otherwise use default
    input_data_path = None
    if request and request.input_data_path and request.input_data_path != "string":
        input_data_path = request.input_data_path
 
    try:
        # Submit the training job to Vertex AI
        if input_data_path:
            result = submit_training_job(ticker=ticker, input_data_path=input_data_path)
        else:
            result = submit_training_job(ticker=ticker)
 
        logger.info(f"Training job submitted for {ticker}: {result['job_name']}")
        return TrainResponse(**result)
 
    except Exception as e:
        logger.error(f"Error submitting training job for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit training job: {str(e)}"
        )
