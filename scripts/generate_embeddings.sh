#!/usr/bin/env bash
set -e

# Clear the data directory
rm -rf data/prod

s3_prefix="s3://bcgl-public-bucket"
data_dir="test-serving" # 'test-serving', 'dev-serving', or 'prod-serving'

# Run the embeddings pipeline
poetry run python scripts/pipeline/run_embedding_pipeline.py --num_pages_to_process 1 \
    --batch_size 100 --bucket_name 'bcgl-public-bucket' --pdf_dir 'archive-small/PDFs/' \
    --remote_data_dir "$data_dir/" --text_model_type 'Dummy' --visual_model_type 'Dummy' --do_text_embedding 1 --do_img_embedding 1 --do_metadata_collection 1
