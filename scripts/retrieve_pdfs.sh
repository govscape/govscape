poetry run python3 scripts/data_prep/process_cdxs.py \
  --backend s3 \
  --output_prefix 'cdx'

#poetry run python3 scripts/data_prep/retrieve_pdfs.py \
#  --backend s3 \
#  --cdx_parquet 'cdx/complete_cdx.parquet' \
#  --output_dir 'pdfs'
