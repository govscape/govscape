variable "environment" {
  description = "Short name used to prefix resource names, e.g. \"next\""
  type        = string
}

variable "site_domains" {
  description = "Hostnames in the govscape.net zone served by CloudFront; the first one is the primary name"
  type        = list(string)

  validation {
    condition     = length(var.site_domains) > 0
    error_message = "site_domains must contain at least one hostname."
  }
}

variable "app_image" {
  description = "Docker image of the API server; prefer an immutable sha-* tag. Changing it replaces the instance."
  type        = string
}

variable "instance_type" {
  description = "x86_64 instance type for the API; FAISS indices are held in memory, so size RAM to the serving data"
  type        = string
  default     = "t3.xlarge"
}

variable "root_volume_gb" {
  description = "Root volume size, which holds the Docker image, downloaded indices, and model cache"
  type        = number
  default     = 60
}

variable "serving_data_dir" {
  description = "Serving directory in the source.coop derivative repository, e.g. prod-serving"
  type        = string
  default     = "prod-serving"
}

variable "server_args" {
  description = "Model and index arguments passed to scripts.serving.run_gunicorn"
  type        = string
  default     = "--text_model BGE --visual_model CLIP --keyword_index_type SQLite --vector_index_type Memory --forward_index_type LMDB"
}

variable "gunicorn_workers" {
  type    = number
  default = 4
}

variable "create_github_oidc_provider" {
  description = "Create the account-wide GitHub Actions OIDC provider; set false if the account already has one"
  type        = bool
  default     = true
}
