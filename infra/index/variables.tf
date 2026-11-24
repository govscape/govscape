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

variable "ivfpq_args" {
  description = "Extra arguments for scripts/indexing/build_ivfpq_index.py, e.g. \"--nlist 65536\""
  type        = string
  default     = ""
}

variable "instance_type" {
  description = "Instance type for the keyword and metadata builders, unless instance_types overrides it"
  type        = string
  default     = "r7i.2xlarge"
}

variable "vector_instance_type" {
  description = "Instance type for the vector (txt, img_pg) builders, unless instance_types overrides it; FAISS trains and assigns on its GPU, and PQ-encodes on its CPUs"
  type        = string
  default     = "g7.8xlarge"
}

variable "vector_root_volume_gb" {
  description = "Root volume size for the vector builders, which build their whole index on it: about 2.2 KB per vector for text (1024-d) and 1.7 KB for page images (768-d), ~2.6 TB and ~2 TB for 1.17B pages"
  type        = number
  default     = 3500
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
  description = "Root volume size for the keyword and metadata builders; it holds the app image, their input and their index"
  type        = number
  default     = 200
}
