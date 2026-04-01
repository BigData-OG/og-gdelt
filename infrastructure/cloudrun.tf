locals {
  registry_name = "og-gdelt-api"
  cloud_run_anme = "${local.name_prefix}-cloudrun-api"
}

data "google_artifact_registry_repository" "gdelt_api" {
  location      = var.gcp_region
  repository_id = local.registry_name
}

data "google_artifact_registry_docker_image" "gdelt_api" {
  location      = data.google_artifact_registry_repository.gdelt_api.location
  repository_id = data.google_artifact_registry_repository.gdelt_api.repository_id
  image_name    = var.latest_cloudrun_image_name
}

resource "google_cloud_run_v2_service" "gdelt_api" {
  name     = local.cloud_run_anme
  location = var.gcp_region
  deletion_protection = false
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    health_check_disabled = true
    containers {
      image = data.google_artifact_registry_docker_image.gdelt_api.self_link
      env {
        name  = "FIRESTORE_DB_NAME"
        value = google_firestore_database.ml_models.name
      }
      env {
        name  = "FIRESTORE_COLLECTION_NAME"
        value = "gdeltModels"
      }
      env {
        name = "GCP_REGION"
        value = var.gcp_region
      }
      env {
        name = "PROJECT_ID"
        value = var.project_id
      }
      resources {
        cpu_idle = true
        limits = {
          cpu    = "2"
          memory = "1024Mi"
        }
      }
      ports {
        container_port = 80
      }
    }
  }
}