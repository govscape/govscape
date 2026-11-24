variable "remote_data_dir" {
  description = "Serving directory in the derivative repository whose indexes to build, e.g. prod-serving/"
  type        = string
}

variable "app_image" {
  description = "Docker image to run; prefer an immutable sha-* tag. Changing it replaces the builder."
  type        = string
}

variable "indexes" {
  description = "Indexes to build, each on its own instance, concurrently: any of txt (text vectors), img_pg (page image vectors), keyword and metadata"
  type        = list(string)
  default     = ["txt", "img_pg", "keyword", "metadata"]

  validation {
    condition     = length(var.indexes) > 0 && alltrue([for index in var.indexes : contains(["txt", "img_pg", "keyword", "metadata"], index)])
    error_message = "indexes must be a non-empty list of txt, img_pg, keyword and metadata."
  }
}

variable "keyword_index_type" {
  description = "Keyword index to build: Tantivy, SQLite, Whoosh, LanceDB or Lucene"
  type        = string
  default     = "Tantivy"
}

variable "forward_index_type" {
  description = "Forward index stored with each vector index: LMDB or SQLite"
  type        = string
  default     = "LMDB"
}

variable "instance_type" {
  description = "Instance type for each builder, unless instance_types overrides it; the builds are CPU and memory bound, and the vector indexes are built in memory"
  type        = string
  default     = "r7i.2xlarge"
}

variable "instance_types" {
  description = "Instance types for particular indexes, e.g. { txt = \"r7i.8xlarge\" }; the others use instance_type"
  type        = map(string)
  default     = {}
}

variable "availability_zones" {
  description = "Zone names to place the builder in; null uses every zone that offers instance_type"
  type        = list(string)
  default     = null
}

variable "use_spot" {
  description = "Run the builders as persistent spot instances that stop on interruption and resume later"
  type        = bool
  default     = true
}

variable "instance_profile" {
  description = "Existing instance profile whose role source.coop allows to write to the govscape repositories"
  type        = string
  default     = "SourceCoopAccess"
}

variable "root_volume_gb" {
  description = "Root volume size; it holds the app image, the input being indexed and the indexes themselves"
  type        = number
  default     = 200
}
