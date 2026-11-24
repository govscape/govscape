#!/usr/bin/env bash
set -e

# Generates a serving directory in the source.coop derivative repository from the
# PDF archive: page text/images, embeddings and metadata, then all four indices.
# Writing needs source.coop credentials, e.g.:
#   PROFILE=eota-pdf-derivative-data MAX_PDFS=100 DATA_DIR=prod-serving scripts/generate_serving_data.sh
# The embedding pipeline checkpoints under $DATA_DIR/checkpoints/, so a later run
# without MAX_PDFS resumes after the PDFs that were already processed.

data_dir="${DATA_DIR:-prod-serving}" # 'test-serving', 'dev-serving', or 'prod-serving'
profile="${PROFILE:?Set PROFILE to an AWS profile holding source.coop credentials}"
text_model="${TEXT_MODEL:-BGE}"
visual_model="${VISUAL_MODEL:-CLIP}"
max_files=100000000

max_pdfs_args=()
if [[ -n "$MAX_PDFS" ]]; then
    max_pdfs_args=(--max_pdfs "$MAX_PDFS")
fi

# Clear the data directory
rm -rf data/prod

poetry run python scripts/pipeline/run_embedding_pipeline.py --remote_data_dir "$data_dir/" \
    --profile "$profile" "${max_pdfs_args[@]}" --batch_size 100 \
    --text_model_type "$text_model" --visual_model_type "$visual_model" \
    --do_text_embedding 1 --do_img_embedding 1 --do_metadata_collection 1

for embedding_type in txt img_pg; do
    rm -rf data/prod
    poetry run python scripts/indexing/generate_index_embedding.py --remote_data_dir "$data_dir" \
        --profile "$profile" --num_pages_to_process $max_files \
        --embedding_type $embedding_type --forward_index_type LMDB
done

rm -rf data/prod
poetry run python scripts/indexing/generate_index_keyword.py --remote_data_dir "$data_dir" \
    --profile "$profile" --num_pages_to_process $max_files --keyword_index_type Tantivy

rm -rf data/prod
poetry run python scripts/indexing/generate_index_metadata.py --remote_data_dir "$data_dir" \
    --profile "$profile" --num_pages_to_process $max_files
