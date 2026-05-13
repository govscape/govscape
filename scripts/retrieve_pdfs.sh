poetry run python3 scripts/data_prep/process_cdxs.py \
  --backend s3 \
  --bucket_name 'bcgl-public-bucket' \
  --output_prefix 'archive/CDX'

#poetry run python3 scripts/data_prep/retrieve_pdfs.py --bucket 'bcgl-public-bucket' --cdx_parquet 'data/cdx_dir/pdf_metadata.parquet' --output_dir 'archive/2020/PDFs'
