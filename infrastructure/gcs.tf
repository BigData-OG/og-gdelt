locals {
  main_data_bucket_name = "${var.project_name}-main-data-${var.environment}"

}

resource "google_storage_bucket" "main_data" {
  name                        = local.main_data_bucket_name
  location                    = "US"
  force_destroy               = true
  public_access_prevention    = "enforced"
  uniform_bucket_level_access = true
  hierarchical_namespace {
    enabled = true
  }

}