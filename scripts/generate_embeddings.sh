#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://bcgl-public-bucket"
data_dir="dev-serving" # 'dev-serving' or 'prod-serving'

# Run the embeddings pipeline
poetry run python scripts/python_helpers/s3_embedding_pipeline.py --num_pages_to_process 100000 \
    --batch_size 100000 --bucket_name 'bcgl-public-bucket' --pdf_dir 'clean-archive/2020/PDFs/' \
    --data_dir "$data_dir/" --model_type 'BGE' --do_text_embedding 0 --do_img_embedding 0 --do_metadata_collection 1

