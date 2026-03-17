# dataproc.tf
resource "google_dataproc_cluster" "school_project_cluster" {
  name   = "${local.name_prefix}-cluster"
  region = var.gcp_region

  # Applies the tags defined in locals.tf (Project, Environment, ManagedBy)
  labels = local.common_tags

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

    # same master/worker node, disable if more than one node is required.
    software_config {
      override_properties = {
        "dataproc:dataproc.allow.zero.workers" = "true"
      }
    }

    # Links to the custom network created in network.tf
    gce_cluster_config {
      subnetwork = google_compute_subnetwork.custom_subnet.id
    }
  }
}
