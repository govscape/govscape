#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://bcgl-public-bucket"
data_dir="test-serving"
index_type="SQLite" # or "Whoosh"

# Run the embeddings pipeline
poetry run python scripts/indexing/generate_index_keyword.py --num_pages_to_process 60000000 --batch_size 1000000 \
                        --bucket_name 'bcgl-public-bucket' --remote_data_dir $data_dir \
                        --keyword_index_type $index_type
