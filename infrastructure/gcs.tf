locals {
  main_data_bucket_name = "${var.project_name}-main-data-${var.environment}"

}

resource "google_storage_bucket" "main_data" {
  name                        = local.main_data_bucket_name
  location                    = "US-WEST1"
  public_access_prevention    = "enforced"
  uniform_bucket_level_access = true
  hierarchical_namespace {
    enabled = true
  }
}

resource "google_storage_bucket_object" "model_training_script" {
  name   = "scripts/training.py"
  source = "${path.module}/../scripts/spark_train_gdelt.py"
  bucket = google_storage_bucket.main_data.name
}