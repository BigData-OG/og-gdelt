"""
Training pipeline - submits Vertex AI custom training jobs and deploys models.
 
This module handles:
1. Submitting training jobs to Vertex AI
2. Polling training job status until completion
3. Registering trained models in the Model Registry
4. Creating endpoints and deploying models for inference
"""
 
import os
from google.cloud.aiplatform_v1 import (
    JobServiceClient,
    CustomJob,
    CustomJobSpec,
    WorkerPoolSpec,
    PythonPackageSpec,
    MachineSpec,
    DiskSpec,
)
from google.cloud import aiplatform, firestore
import logging
import time

from api.services.model_repository import ModelRepository
 
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
 
# The pre-built sklearn container for SERVING predictions (from previous config)
SERVING_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-6:latest"
 
# Machine configuration
MACHINE_TYPE = "e2-standard-4"  # For training
SERVING_MACHINE_TYPE = "n1-standard-4"  # For inference (from previous endpoint config)
 
# Service account
SERVICE_ACCOUNT = f"{PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
 
# Deployment settings (from previous aramco endpoint config)
MIN_REPLICAS = 1
MAX_REPLICAS = 2
TRAFFIC_SPLIT = {0: 100}  # 100% traffic to the deployed model
 
 
# ============================================================
# STEP 1: Submit Training Job (existing code, unchanged)
# ============================================================
 
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
 
 
# ============================================================
# STEP 2: Wait for Training to Complete
# ============================================================
 
def wait_for_training_job(job_name: str, poll_interval: int = 60) -> str:
    """
    Poll the training job until it finishes.
    
    This runs in the background — the user doesn't see this happening.
    It checks the job status every poll_interval seconds.
    
    Args:
        job_name: Full resource name of the job 
                  (e.g., "projects/.../locations/.../customJobs/12345")
        poll_interval: How often to check status, in seconds (default 60)
    
    Returns:
        Final job state as a string ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", etc.)
    """
    client = JobServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )
 
    logger.info(f"Waiting for training job to complete: {job_name}")
 
    while True:
        job = client.get_custom_job(name=job_name)
        state = job.state.name  # e.g., "JOB_STATE_SUCCEEDED"
 
        logger.info(f"  Training job state: {state}")
 
        if state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            return state
 
        time.sleep(poll_interval)
 
 
# ============================================================
# STEP 3: Register Model in Vertex AI Model Registry
# ============================================================
 
def register_model(ticker: str) -> aiplatform.Model:
    """
    Upload/register the trained model to Vertex AI Model Registry.
    
    This tells Vertex AI: "here's a model file in GCS, and here's the
    container that knows how to load and serve it."
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        The registered Model object
    """
    safe_name = ticker.lower().replace(".", "-").replace("/", "-")
    
    # This matches the path pattern from your training output + save_model function
    # train_results/vertex/model/{ticker}/ is the output_dir
    # Inside that, save_model creates model/model.joblib
    # So the artifact directory is: train_results/vertex/model/{ticker}/model/
    artifact_uri = f"gs://{BUCKET_NAME}/train_results/vertex/model/{safe_name}/model/"
    
    model_display_name = f"{safe_name}-pred"
 
    logger.info(f"Registering model: {model_display_name}")
    logger.info(f"  Artifact URI: {artifact_uri}")
    logger.info(f"  Serving container: {SERVING_CONTAINER_URI}")
 
    # Initialize the high-level SDK (we use it here because model upload
    # is simpler with it — no async issues like with training submission)
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}",
    )
 
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=SERVING_CONTAINER_URI,
    )
 
    logger.info(f"Model registered successfully!")
    logger.info(f"  Model resource name: {model.resource_name}")
    
    return model
 
 
# ============================================================
# STEP 4: Create or Get Endpoint
# ============================================================
 
def get_or_create_endpoint(ticker: str) -> aiplatform.Endpoint:
    """
    Find an existing endpoint for this ticker, or create a new one.
    
    Previously used convention: one endpoint per company, named after
    the ticker (e.g., "aramco", "amzn", "tsla").
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        The Endpoint object
    """
    safe_name = ticker.lower().replace(".", "-").replace("/", "-")
 
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}",
    )
 
    # Check if an endpoint already exists for this ticker
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{safe_name}"',
        order_by="create_time desc",
    )
 
    if endpoints:
        endpoint = endpoints[0]
        logger.info(f"Found existing endpoint: {endpoint.display_name} ({endpoint.resource_name})")
        return endpoint
 
    # No existing endpoint — create a new one
    logger.info(f"Creating new endpoint: {safe_name}")
    endpoint = aiplatform.Endpoint.create(
        display_name=safe_name,
        # Standard access, no dedicated DNS (matching previous config)
    )
 
    logger.info(f"Endpoint created: {endpoint.resource_name}")
    return endpoint
 
 
# ============================================================
# STEP 5: Deploy Model to Endpoint
# ============================================================
 
def deploy_model_to_endpoint(
    model: aiplatform.Model,
    endpoint: aiplatform.Endpoint,
) -> None:
    """
    Deploy a registered model to an endpoint.
    
    THIS IS THE SLOW STEP (~15-20 minutes). Google is provisioning VMs
    to run the prediction server.
    
    Settings match previous aramco deployment:
    - n1-standard-4 machine
    - Min 1 replica, Max 2 replicas
    - 100% traffic
    - No GPU
    - Access logging enabled
    
    Args:
        model: The registered Model from step 3
        endpoint: The Endpoint from step 4
    """
    logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.display_name}")
    logger.info(f"  Machine type: {SERVING_MACHINE_TYPE}")
    logger.info(f"  Replicas: {MIN_REPLICAS}-{MAX_REPLICAS}")
    logger.info("  This will take ~15-20 minutes...")
 
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=model.display_name,
        machine_type=SERVING_MACHINE_TYPE,
        min_replica_count=MIN_REPLICAS,
        max_replica_count=MAX_REPLICAS,
        traffic_split={"0": 100},
        enable_access_logging=True,
    )
 
    logger.info(f"Model deployed successfully!")
    logger.info(f"  Endpoint ID: {endpoint.resource_name}")
 
 
# ============================================================
# FULL BACKGROUND PIPELINE: Wait → Register → Deploy
# ============================================================
 
# This dict tracks the status of all jobs so the /train/status endpoint can report it
job_tracker = {}
 
def train_and_deploy_background(job_name: str, ticker: str):
    """
    Background task that runs after /train returns to the user.
    
    1. Polls the training job until it finishes
    2. Registers the model in Model Registry
    3. Creates/finds an endpoint
    4. Deploys the model to the endpoint
    
    The user can check progress via /train/status/{ticker}
    
    Args:
        job_name: Full Vertex AI job resource name
        ticker: Stock ticker symbol
    """
    safe_name = ticker.lower().replace(".", "-").replace("/", "-")
 
    try:
        # --- Phase 1: Wait for training ---
        job_tracker[safe_name] = {
            "status": "training",
            "message": "Training job is running on Vertex AI...",
            "ticker": ticker.upper(),
        }
 
        final_state = wait_for_training_job(job_name, poll_interval=60)
 
        if final_state != "JOB_STATE_SUCCEEDED":
            job_tracker[safe_name] = {
                "status": "failed",
                "message": f"Training job failed with state: {final_state}",
                "ticker": ticker.upper(),
            }
            logger.error(f"Training failed for {ticker}: {final_state}")
            return
 
        # --- Phase 2: Register model ---
        job_tracker[safe_name] = {
            "status": "registering_model",
            "message": "Training complete! Registering model in Vertex AI Model Registry...",
            "ticker": ticker.upper(),
        }
 
        model = register_model(ticker)
 
        # --- Phase 3: Create/find endpoint ---
        job_tracker[safe_name] = {
            "status": "creating_endpoint",
            "message": "Model registered! Creating prediction endpoint...",
            "ticker": ticker.upper(),
        }
 
        endpoint = get_or_create_endpoint(ticker)
 
        # --- Phase 4: Deploy model ---
        job_tracker[safe_name] = {
            "status": "deploying",
            "message": "Deploying model to endpoint (this takes ~15-20 min)...",
            "ticker": ticker.upper(),
        }
 
        deploy_model_to_endpoint(model, endpoint)
        firestore_client = firestore.Client(project=PROJECT_ID,database=os.environ.get("FIRESTORE_DB_NAME", "og-gdelt-dev-firestore-db"))
        model_repository = ModelRepository(firestore_client)
        model_repository.save_model_id(ticker, endpoint.resource_name)
        # --- Done! ---
        job_tracker[safe_name] = {
            "status": "deployed",
            "message": f"Model for {ticker.upper()} is live and ready for predictions!",
            "ticker": ticker.upper(),
            "endpoint_id": endpoint.resource_name,
            "model_id": model.resource_name,
        }
 
        logger.info(f"Full pipeline complete for {ticker}!")
 
    except Exception as e:
        job_tracker[safe_name] = {
            "status": "error",
            "message": f"Pipeline error: {str(e)}",
            "ticker": ticker.upper(),
        }
        logger.error(f"Background pipeline failed for {ticker}: {e}")