resource "google_compute_network" "custom_vpc" {
  name                    = "${local.name_prefix}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "custom_subnet" {
  name                     = "${local.name_prefix}-subnet"
  ip_cidr_range            = "10.0.1.0/24"
  region                   = var.gcp_region
  network                  = google_compute_network.custom_vpc.id
  private_ip_google_access = true
}

# Firewall rule to allow all internal traffic within the VPC network.
# This is crucial for Dataproc master and worker nodes to communicate.
resource "google_compute_firewall" "allow_internal" {
  name    = "${local.name_prefix}-allow-internal"
  network = google_compute_network.custom_vpc.name

  allow {
    protocol = "icmp"
  }
  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  source_ranges = [google_compute_subnetwork.custom_subnet.ip_cidr_range]
}

# removed because of risky external config.
# Firewall rule to allow external access for SSH, RDP, and ICMP.
# resource "google_compute_firewall" "allow_external_access" {
#   name    = "${local.name_prefix}-allow-external"
#   network = google_compute_network.custom_vpc.name
#
#   allow {
#     protocol = "tcp"
#     ports    = ["22", "3389"] # SSH and RDP
#   }
#   allow {
#     protocol = "icmp"
#   }
#   # WARNING: This allows traffic from any source. For production, you should
#   # restrict this to a specific IP range, e.g., ["YOUR_IP_ADDRESS/32"].
#   source_ranges = ["0.0.0.0/0"]
# }
