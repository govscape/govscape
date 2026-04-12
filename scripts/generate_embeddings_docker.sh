#!/bin/bash

# This script is a sample of starting the embedding processs using a docker image.

# Build the docker image first using
# $ docker build -t govscape .

docker run -v ${LOCAL_WORKSPACE_FOLDER:-.}/data:/data --ipc=host --rm -i --gpus all govscape bash <<'EOF'

# Embedding with GPU

poetry run python scripts/pipeline/run_embedding_pipeline.py --num_pages_to_process 5 \
    --batch_size 100 --backend 'local' --local_base_dir '/data/s3_mock' --pdf_dir 'archive/PDFs/' \
    --remote_data_dir "test-serving" --text_model_type 'BGE' --visual_model_type 'CLIP'

# The command above can be modified for different parameters.
EOF
