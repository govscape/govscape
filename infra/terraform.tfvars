environment  = "next"
site_domains = ["next.govscape.net"]

# Pin to an immutable tag from the publish-app-image workflow (sha-<short sha>);
# applying a new tag replaces the API instance.
app_image = "ghcr.io/govscape/govscape:sha-24831ba"

# The 10,000-PDF ingestion pilot, embedded with Qwen3 (text) and SigLIP 2
# (images); queries must be embedded with the same models.
serving_data_dir = "pilot-ocr-run7-serving"
server_args      = "--text_model Qwen3 --visual_model SigLIP --keyword_index_type Tantivy --vector_index_type Memory --forward_index_type LMDB"
instance_type    = "m6id.xlarge"
