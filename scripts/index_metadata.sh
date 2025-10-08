#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://bcgl-public-bucket"
data_dir="dev-serving"

# Run the embeddings pipeline
poetry run python scripts/python_helpers/generate_index_metadata.py --bucket_name 'bcgl-public-bucket' --cdx_parquet_key 'archive/2020/CDX/pdf_metadata.parquet' --metadata_prefix $data_dir/metadata --output_prefix 'dev-serving/index_metadata' --output_dir 'data/index_metadata' --num_pages_to_process 100000

