#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://bcgl-public-bucket/"
data_dir="dev-serving/"

# Download the indices
s5cmd sync $s3_prefix/$data_dir/index/* data/prod/index
s5cmd sync $s3_prefix/$data_dir/index_img/* data/prod/index_img
s5cmd sync $s3_prefix/$data_dir/index_keyword/* data/prod/index_keyword

# Run the embeddings pipeline
poetry run python scripts/s3_ec2_embedding_pipeline.py --num_pages_to_process 5 --bucket_name 'bcgl-public-bucket' --pdf_dir 'archive/2020/PDFs/' --data_dir $data_dir --model_type 'BGE'

