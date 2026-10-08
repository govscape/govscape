#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://govscape/eota-derivative-data"
data_dir="test-serving"

# Run the embeddings pipeline
poetry run python scripts/indexing/generate_index_metadata.py --remote_data_dir $data_dir --num_pages_to_process 50
