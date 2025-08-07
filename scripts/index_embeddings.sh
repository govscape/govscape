#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://bcgl-public-bucket"
data_dir="dev-serving"
embedding_type="/embedding"

# Download the indices
s5cmd sync $s3_prefix/$data_dir/index/* data/prod/index

# Run the embeddings pipeline
poetry run python scripts/python_helpers/s3_embedding_indexing_pipeline.py --num_pages_to_process 5 --bucket_name 'bcgl-public-bucket' --embedding_prefix "$embedding_type" --in_data_dir $data_dir --out_data_dir $data_dir
