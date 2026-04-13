#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

data_dir="test-serving"

# Run the embeddings pipeline
poetry run python scripts/indexing/generate_index_embedding.py --num_pages_to_process 1000000 --bucket_name 'bcgl-public-bucket' --embedding_type img_pg --remote_data_dir $data_dir
