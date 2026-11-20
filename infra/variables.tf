variable "region" {
  description = "Region for the API instance, its network, and the site bucket"
  type        = string
  default     = "us-east-2"
}

variable "environment" {
  description = "Short name used to prefix resource names, e.g. \"next\""
  type        = string
}

variable "zone_name" {
  description = "Existing Route 53 public hosted zone that site_domains live in"
  type        = string
  default     = "govscape.net"
}

variable "site_domains" {
  description = "Hostnames served by the CloudFront distribution; the first one is the primary name"
  type        = list(string)

  validation {
    condition     = length(var.site_domains) > 0
    error_message = "site_domains must contain at least one hostname."
  }
}

variable "site_bucket_name" {
  description = "Globally unique name of the frontend bucket (default: govscape-<environment>-site)"
  type        = string
  default     = null
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

variable "gunicorn_threads" {
  type    = number
  default = 4
}

variable "vpc_cidr" {
  type    = string
  default = "10.42.0.0/16"
}

variable "github_repo" {
  description = "GitHub repository (owner/name) allowed to deploy the frontend"
  type        = string
  default     = "govscape/govscape"
}

variable "github_deploy_branch" {
  description = "Branch whose workflows may assume the deploy role"
  type        = string
  default     = "main"
}

variable "create_github_oidc_provider" {
  description = "Create the account-wide GitHub Actions OIDC provider; set false if the account already has one"
  type        = bool
  default     = true
}
