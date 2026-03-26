variable "project_name" {
  description = "Name of the project used to prefix resource names"
  type        = string
  default     = "og-gdelt"
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "gdelt-stock-sentiment-analysis"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}


variable "gcp_region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-west1"
}

variable "training_data_path" {
  description = "GCS path to the training data"
  type        = string
  default     = "cleaned_data/combined_data_clean_000000000000.csv"
}