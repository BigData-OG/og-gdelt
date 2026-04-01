
resource "google_firestore_database" "ml_models" {
  project                           = var.project_id
  name                              = "${local.name_prefix}-firestore-db"
  location_id                       = "nam5"
  type                              = "FIRESTORE_NATIVE"
  delete_protection_state           = "DELETE_PROTECTION_ENABLED"
  deletion_policy                   = "DELETE"
}

resource "google_firestore_field" "company_name" {
  project    = var.project_id
  database   = google_firestore_database.ml_models.name
  collection = "gdeltModels"
  field      = "companyName"

  index_config {}
}

resource "google_firestore_field" "model_id" {
  project    = var.project_id
  database   = google_firestore_database.ml_models.name
  collection = "gdeltModels"
  field      = "modelId"

  index_config {}
}