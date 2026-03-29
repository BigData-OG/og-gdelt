import argparse
import logging
from google.cloud import dataproc_v1 as dataproc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def submit_pyspark_job(project_id, region, cluster_name, job_file_path, *args):
    """
    Submits a Pyspark job to a Dataproc cluster.
    """
    logger.info(f"Submitting PySpark job to cluster '{cluster_name}' in region '{region}'...")
    
    # Create the job client
    job_client = dataproc.JobControllerClient(
        client_options={"api_endpoint": f"{region}-dataproc.googleapis.com:443"}
    )

    # Define the PySpark job specification
    job = {
        "placement": {"cluster_name": cluster_name},
        "pyspark_job": {
            "main_python_file_uri": job_file_path,
            "args": args,
        },
    }

    try:
        # Submit the job
        operation = job_client.submit_job_as_operation(
            request={"project_id": project_id, "region": region, "job": job}
        )
        logger.info("Waiting for job to complete...")
        
        # Wait for the job to complete
        response = operation.result()
        logger.info(f"Job finished successfully. Output detailed in Dataproc logs.")
        logger.info(f"Job ID: {response.reference.job_id}")
        
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit a PySpark job to Dataproc.")
    parser.add_argument("--project_id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--region", required=True, help="Compute region of the cluster")
    parser.add_argument("--cluster_name", required=True, help="Name of the Dataproc cluster")
    parser.add_argument("--job_file_path", required=True, help="GCS URI of the PySpark job file (gs://...)")
    parser.add_argument("job_args", nargs="*", help="Optional arguments to pass to the PySpark job")

    args = parser.parse_args()

    submit_pyspark_job(
        args.project_id,
        args.region,
        args.cluster_name,
        args.job_file_path,
        *args.job_args
    )
