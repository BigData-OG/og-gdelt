"""
Training pipeline - submits Vertex AI custom training jobs.
 
This module handles submitting training jobs to Vertex AI using the
existing training package (training_package.tar.gz) in GCS.
"""
 
from google.cloud.aiplatform_v1 import (
    JobServiceClient,
    CustomJob,
    CustomJobSpec,
    WorkerPoolSpec,
    PythonPackageSpec,
    MachineSpec,
    DiskSpec,
)
import logging
 
logger = logging.getLogger(__name__)
 
# --- Configuration ---
PROJECT_ID = "gdelt-stock-sentiment-analysis"
REGION = "us-west1"
BUCKET_NAME = "og-gdelt-main-data-dev"
PROJECT_NUMBER = "41778621396"
 
# Training package location in GCS
TRAINING_PACKAGE_URI = f"gs://{BUCKET_NAME}/scripts/training_package.tar.gz"
PYTHON_MODULE = "trainer.train"
 
# The pre-built sklearn container that Vertex AI uses to run the training
CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest"
 
# Machine configuration
MACHINE_TYPE = "e2-standard-4"
 
# Service account
SERVICE_ACCOUNT = f"{PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
 
 
def submit_training_job(
    ticker: str,
    input_data_path: str = "cleaned_data/combined_data_clean_000000000000.csv",
) -> dict:
    """
    Submit a Vertex AI custom training job for a given ticker.
 
    Args:
        ticker: Stock ticker symbol (e.g., "AMZN", "TSLA", "GOOGL")
        input_data_path: Path to training data inside the GCS bucket
 
    Returns:
        dict with job_id, job_name, and status
    """
 
    # Clean ticker for use in names (replace dots/special chars)
    safe_name = ticker.lower().replace(".", "-").replace("/", "-")
    display_name = f"train-{safe_name}-custom-job"
    output_dir = f"gs://{BUCKET_NAME}/train_results/vertex/model/{safe_name}/"
    training_args = [ticker, BUCKET_NAME, input_data_path]
 
    logger.info(f"Submitting training job: {display_name}")
    logger.info(f"  Ticker: {ticker}")
    logger.info(f"  Data path: gs://{BUCKET_NAME}/{input_data_path}")
    logger.info(f"  Output dir: {output_dir}")
 
    try:
        client = JobServiceClient(
            client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
        )
 
        custom_job = CustomJob(
            display_name=display_name,
            job_spec=CustomJobSpec(
                worker_pool_specs=[
                    WorkerPoolSpec(
                        machine_spec=MachineSpec(machine_type=MACHINE_TYPE),
                        replica_count=1,
                        disk_spec=DiskSpec(
                            boot_disk_type="pd-standard",
                            boot_disk_size_gb=100,
                        ),
                        python_package_spec=PythonPackageSpec(
                            executor_image_uri=CONTAINER_URI,
                            package_uris=[TRAINING_PACKAGE_URI],
                            python_module=PYTHON_MODULE,
                            args=training_args,
                        ),
                    )
                ],
                service_account=SERVICE_ACCOUNT,
                base_output_directory={"output_uri_prefix": output_dir},
            ),
        )
 
        parent = f"projects/{PROJECT_ID}/locations/{REGION}"
        response = client.create_custom_job(parent=parent, custom_job=custom_job)
 
        job_name = response.name
        job_id = job_name.split("/")[-1]
 
        logger.info(f"Training job submitted successfully!")
        logger.info(f"  Job name: {job_name}")
        logger.info(f"  Job ID: {job_id}")
        logger.info(
            f"  View at: https://console.cloud.google.com/ai/platform/locations/{REGION}/training/{job_id}?project={PROJECT_NUMBER}"
        )
 
        return {
            "status": "training_started",
            "job_name": job_name,
            "display_name": display_name,
            "ticker": ticker.upper(),
            "data_path": f"gs://{BUCKET_NAME}/{input_data_path}",
            "output_dir": output_dir,
        }
 
    except Exception as e:
        logger.error(f"Failed to submit training job: {e}")
        raise