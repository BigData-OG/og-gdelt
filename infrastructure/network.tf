# network.tf
resource "google_compute_network" "custom_vpc"{
  name = "${local.name_prefix}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "custom_subnet" {
  name = "${local.name_prefix}-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region = var.gcp_region
  network = google_compute_network.custom_vpc.id
  private_ip_google_access = true
}
