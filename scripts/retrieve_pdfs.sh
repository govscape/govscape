poetry run python3 scripts/data_prep/process_cdxs.py --bucket 'eotarchive' --cdx_file_paths 'data/cdx_dir/cdx.paths' --output_dir 'data/cdx_dir' --output_prefix 'archive/2020'

#poetry run python3 scripts/data_prep/retrieve_pdfs.py --bucket 'bcgl-public-bucket' --cdx_parquet 'data/cdx_dir/pdf_metadata.parquet' --output_dir 'archive/2020/PDFs'
