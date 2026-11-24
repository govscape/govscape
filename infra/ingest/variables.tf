variable "num_servers" {
  description = "Number of worker instances; each one processes every num_servers-th shard of the PDFs"
  type        = number

  validation {
    condition     = var.num_servers >= 1
    error_message = "num_servers must be at least 1."
  }
}

variable "remote_data_dir" {
  description = "Directory in the derivative repository to write to, e.g. dev-serving/"
  type        = string
}

variable "app_image" {
  description = "Docker image to run; prefer an immutable sha-* tag. Changing it replaces the workers."
  type        = string
}

variable "instance_type" {
  description = "Instance type with one GPU; the pipeline uses one GPU and all of the instance's CPUs"
  type        = string
  default     = "g6.2xlarge"
}

variable "availability_zones" {
  description = "Zone names to place workers in, e.g. those with spot capacity for instance_type (`aws ec2 get-spot-placement-scores`); null uses every zone that offers it"
  type        = list(string)
  default     = null
}

variable "use_spot" {
  description = "Run workers as persistent spot instances that stop on interruption and resume later"
  type        = bool
  default     = true
}

variable "max_pdfs" {
  description = "Stop each worker after this many PDFs, e.g. for a pilot run; null processes all of its shards"
  type        = number
  default     = null
}

variable "pipeline_args" {
  description = "Extra arguments for scripts/pipeline/run_embedding_pipeline.py"
  type        = string
  default     = "--text_model_type BGE --visual_model_type CLIP"
}

variable "vllm_image" {
  description = "vLLM image to serve Qwen3 text embeddings next to the pipeline, e.g. vllm/vllm-openai:v0.34.0; use with --text_model_type Qwen3vLLM in pipeline_args. null runs no vLLM server."
  type        = string
  default     = null
}

variable "vllm_gpu_memory_utilization" {
  description = "On single-GPU instances, the fraction of GPU memory the vLLM server may use; the pipeline's image models need the rest. On multi-GPU instances vLLM gets every GPU but the first to itself."
  type        = number
  default     = 0.5
}

variable "vllm_extra_args" {
  description = "Extra arguments for `vllm serve`, e.g. \"--max-num-batched-tokens 32768\""
  type        = string
  default     = ""
}

variable "instance_profile" {
  description = "Existing instance profile whose role source.coop allows to write to the govscape repositories"
  type        = string
  default     = "SourceCoopAccess"
}

variable "root_volume_gb" {
  description = "Root volume size. The AMI and the app and vLLM images take ~85 GB, and up to ~6 batches of a few GB each wait on disk at once"
  type        = number
  default     = 200
}
