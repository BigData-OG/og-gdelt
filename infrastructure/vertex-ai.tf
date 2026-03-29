resource "local_file" "dynamic_ml_config" {
  content = jsonencode({
    bucket = google_storage_bucket.main_data.name,
    data_path = var.training_data_path,
    project_id       = var.project_id
  })
  filename = "${path.module}/ml/trainer/config.json"
}

resource "null_resource" "packaged_ml_code" {
  depends_on = [local_file.dynamic_ml_config]
  triggers = {
    key = filemd5("${path.module}/ml/trainer/train.py") # Re-run if the training script changes
  }
  provisioner "local-exec" {
    command = "cd ${path.module} && python setup.py sdist --formats=gztar"
  }
}


resource "google_storage_bucket_object" "vertex_ai_model_training_package" {
  depends_on = [null_resource.packaged_ml_code, google_storage_bucket.main_data]
  name   = "scripts/training_package.tar.gz"
  source = "${path.module}/dist/trainer-0.1.tar.gz"
  bucket = google_storage_bucket.main_data.name
  lifecycle {
    replace_triggered_by = [ null_resource.packaged_ml_code ]
  }
}