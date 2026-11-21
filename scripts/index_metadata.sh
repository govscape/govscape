#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

data_dir="test-serving"

# Run the embeddings pipeline
poetry run python scripts/indexing/generate_index_metadata.py --remote_data_dir $data_dir --num_pages_to_process 50
