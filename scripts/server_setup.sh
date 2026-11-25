#!/usr/bin/env bash
set -e

# Server setup: run_gunicorn downloads the indices from the serving directory in the
# source.coop derivative repository (anonymously) before starting gunicorn.
remote_data_dir="prod-serving" # 'dev-serving', 'test-serving', or 'prod-serving'
local_data_dir="data/prod"

poetry run python -m scripts.serving.run_gunicorn --backend s3 \
    --remote_data_directory "$remote_data_dir" --local_data_directory "$local_data_dir" \
    --text_model BGE --visual_model CLIP --keyword_index_type Tantivy \
    --vector_index_type Memory
