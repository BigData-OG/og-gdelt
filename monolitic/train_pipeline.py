import logging
import time
from train import train_local_model  # Importing the local training function you built

logger = logging.getLogger(__name__)

# This dict tracks the status of all jobs so the /train/status endpoint can report it
job_tracker = {}


def train_and_deploy_background(ticker: str, data_path: str):
    """
    Background task that runs after /train returns to the user.

    1. Updates the tracker to 'training'
    2. Executes the CPU-bound local training script
    3. Updates the tracker to 'deployed' once saved locally

    The user can check progress via /train/status/{ticker}
    """
    safe_name = ticker.lower().replace(".", "-").replace("/", "-")

    try:
        # --- Phase 1: Local Training ---
        job_tracker[safe_name] = {
            "status": "training",
            "message": f"Training model locally in the container for {ticker.upper()}...",
            "ticker": ticker.upper(),
        }
        logger.info(f"Starting background training for {ticker}")

        # Execute the local training script (this will block this specific background thread
        # result = train_local_model(ticker)
        result = train_local_model(ticker, data_path)

        # --- Phase 2: Local "Deployment" ---
        job_tracker[safe_name] = {
            "status": "deploying",
            "message": "Training complete! Model saved locally. Loading into API memory...",
            "ticker": ticker.upper(),
        }

        # Adding a tiny sleep just so if someone polls the status, they might catch the 'deploying' state
        time.sleep(1)

        # --- Phase 3: Done ---
        job_tracker[safe_name] = {
            "status": "deployed",
            "message": f"Model for {ticker.upper()} is live in the container and ready for predictions!",
            "ticker": ticker.upper(),
            "mae": result.get("mae", "N/A"),
        }

        logger.info(f"Full local pipeline complete for {ticker}!")

    except Exception as e:
        job_tracker[safe_name] = {
            "status": "error",
            "message": f"Pipeline error: {str(e)}",
            "ticker": ticker.upper(),
        }
        logger.error(f"Background pipeline failed for {ticker}: {e}")
