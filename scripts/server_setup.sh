#!/usr/bin/env bash
set -e

# Server setup
s3_prefix="s3://bcgl-public-bucket/test-serving"
data_dir="data/prod" # 'dev-serving' or 'prod-serving'
#s5cmd sync $s3_prefix/embeddings/* data/prod/embeddings
#s5cmd sync $s3_prefix/embeddings_img_pg/* data/prod/embeddings_img_pg
#s5cmd sync $s3_prefix/img/* data/prod/img
s5cmd sync $s3_prefix/index/* data/prod/index
s5cmd sync $s3_prefix/index_img_pg/* data/prod/index_img_pg
s5cmd sync $s3_prefix/index_keyword/* data/prod/index_keyword
s5cmd sync $s3_prefix/index_metadata/* data/prod/index_metadata
#s5cmd sync $s3_prefix/metadata/* data/prod/metadata

poetry run python scripts/python_helpers/run_gunicorn.py --data-directory ./data/prod --text_model BGE --visual_model CLIP --keyword_index_type LanceDB --vector_index_type Memory 