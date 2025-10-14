#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://bcgl-public-bucket"
data_dir="test-serving"
embedding_type="/embeddings/"
index_dir="index"

# Download the indices
s5cmd sync $s3_prefix/$data_dir/$index_dir/* data/$data_dir/$index_dir

# Run the embeddings pipeline
poetry run python scripts/python_helpers/generate_index_embedding.py --num_pages_to_process 1000000 --bucket_name 'bcgl-public-bucket' --embedding_prefix "$embedding_type" --in_data_dir $data_dir --out_data_dir $data_dir --out_index_prefix $index_dir 
