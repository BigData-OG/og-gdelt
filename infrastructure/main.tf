terraform {
  backend "gcs" {
    bucket = "og-gdelt-terraform-state-bucket"
    prefix = "terraform/state"
  }
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.20"
    }
  }
  required_version = ">= 1.2"
}


provider "google" {
  project = var.project_id
  region  = var.gcp_region
}


