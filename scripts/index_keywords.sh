#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://bcgl-public-bucket"
data_dir="dev-serving"
index_type="SQLite" # or "Whoosh"

# Download the indices
s5cmd sync $s3_prefix/$data_dir/index_keyword/* data/prod/index_keyword

# Run the embeddings pipeline
poetry run python scripts/python_helpers/generate_index_keyword.py --num_pages_to_process 60000000 --batch_size 1000000 \
                        --bucket_name 'bcgl-public-bucket' --in_data_dir $data_dir --out_data_dir $data_dir \
                        --keyword_index_type $index_type 

