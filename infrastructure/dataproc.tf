# dataproc.tf
locals {
  cluster_name  = "${local.name_prefix}-cluster"
  template_name = "${local.name_prefix}-train-regression-template"
}
resource "google_dataproc_autoscaling_policy" "og_gdelt_autoscaling" {
  policy_id = "${local.name_prefix}-autoscaling-policy"
  location  = var.gcp_region

  worker_config {
    min_instances = 2 # minimum 2 total nodes
    max_instances = 5
  }

  basic_algorithm {
    yarn_config {
      graceful_decommission_timeout = "30s"
      scale_up_factor               = 0.5
      scale_down_factor             = 0.5
    }
  }
}


resource "google_dataproc_workflow_template" "template" {
  name     = local.template_name
  location = var.gcp_region
  placement {
    managed_cluster {
      cluster_name = local.cluster_name
      config {
        # Uses the bucket created in gcs.tf for temporary staging
        staging_bucket = google_storage_bucket.main_data.name
        temp_bucket    = google_storage_bucket.main_data.name

        master_config {
          num_instances = 1 # initial number of master nodes
          machine_type  = "e2-standard-4"
          disk_config {
            boot_disk_size_gb = 30 # Keeps storage costs minimal (GBs)
          }
        }

        worker_config {
          num_instances = 2 # Initial number of worker nodes
          machine_type  = "e2-standard-4"
          disk_config {
            boot_disk_size_gb = 30
          }
        }

        # Links to the custom network created in network.tf
        gce_cluster_config {
          subnetwork = google_compute_subnetwork.custom_subnet.id
        }

        autoscaling_config {
          policy = google_dataproc_autoscaling_policy.og_gdelt_autoscaling.name
        }
      }

    }
  }
  jobs {
    step_id = "train-regression-models"
    pyspark_job {
      main_python_file_uri = google_storage_bucket_object.model_training_script.self_link
      args = [
        google_storage_bucket.main_data.url, # Input data path
        var.training_data_path, # Training data path within the bucket
        "${google_storage_bucket.main_data.url}/regression_models" # Output path for models
      ]
    }
    
  }
}