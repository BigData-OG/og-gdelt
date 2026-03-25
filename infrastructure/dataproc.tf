# dataproc.tf
resource "google_dataproc_autoscaling_policy" "og_gdelt_autoscaling" {
  policy_id = "${local.name_prefix}-autoscaling-policy"
  location    = var.gcp_region

  worker_config {
    min_instances = 1
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

resource "google_dataproc_cluster" "school_project_cluster" {
  name   = "${local.name_prefix}-cluster"
  region = var.gcp_region

  cluster_config {
    # Uses the bucket created in gcs.tf for temporary staging
    staging_bucket = google_storage_bucket.main_data.name
    temp_bucket    = google_storage_bucket.main_data.name

    master_config {
      num_instances = 1
      machine_type  = "e2-standard-2"
      disk_config {
        boot_disk_type    = "pd-standard"
        boot_disk_size_gb = 30 # Keeps storage costs minimal
      }
    }

    worker_config {
      num_instances = 1 # Initial number of worker nodes
      machine_type  = "e2-standard-2"
      disk_config {
        boot_disk_type    = "pd-standard"
        boot_disk_size_gb = 30
      }
    }

    # Links to the custom network created in network.tf
    gce_cluster_config {
      subnetwork = google_compute_subnetwork.custom_subnet.id
    }

    autoscaling_config {
      policy_uri = google_dataproc_autoscaling_policy.og_gdelt_autoscaling.name
    }
  }
}
