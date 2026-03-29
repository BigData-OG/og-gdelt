resource "local_file" "dynamic_ml_config" {
  content = jsonencode({
    bucket = google_storage_bucket.main_data.name,
    data_path = var.training_data_path,
    project_id       = var.project_id
  })
  filename = "${path.module}/ml/trainer/config.json"
}

resource "null_resource" "package_code" {
  depends_on = [local_file.dynamic_ml_config]

  provisioner "local-exec" {
    command = "cd ${path.module}/ml && python setup.py sdist --formats=gztar"
  }
}


resource "google_storage_bucket_object" "vertex_ai_model_training_package" {
  depends_on = [null_resource.package_code, google_storage_bucket.main_data]
  name   = "scripts/training_package.tar.gz"
  source = "${path.module}/ml/dist/gdelt_trainer-0.1.gz"
  bucket = google_storage_bucket.main_data.name
}