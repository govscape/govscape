#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

data_dir="test-serving"
index_type="SQLite" # or "Whoosh"

# Run the embeddings pipeline
poetry run python scripts/indexing/generate_index_keyword.py --num_pages_to_process 60000000 --batch_size 1000000 \
                        --remote_data_dir $data_dir \
                        --keyword_index_type $index_type
