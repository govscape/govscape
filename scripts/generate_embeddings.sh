#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://bcgl-public-bucket"
data_dir="test-serving" # 'test-serving', 'dev-serving', or 'prod-serving'

# Run the embeddings pipeline
poetry run python scripts/python_helpers/run_embedding_pipeline.py --num_pages_to_process 2 \
    --batch_size 10000 --bucket_name 'bcgl-public-bucket' --pdf_dir 'archive-small/PDFs/' \
    --data_dir "$data_dir/" --model_type 'BGE' --do_text_embedding 1 --do_img_embedding 1 --do_metadata_collection 1

