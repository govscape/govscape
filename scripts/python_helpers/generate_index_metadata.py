import subprocess
from govscape.indexing import SQLiteMetadataIndex
import json
import argparse
import pandas as pd
from urllib.parse import urlparse
import time
import os
import multiprocessing
import shutil
from govscape.data_loader import build_data_loader

def extract_subdomain(url):
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return None
    parts = hostname.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return hostname

PROGRESS_PATH = "index_metadata_progress.json"
BATCH_SIZE = 100000
def main():    
    parser = argparse.ArgumentParser(description='Process CDX files from S3.')
    parser.add_argument('--bucket_name', required=True, help='S3 bucket name')
    parser.add_argument('--cdx_parquet_key', required=True, help='S3 Key for CDX parquet file')
    parser.add_argument('--metadata_prefix', required=True, help='S3 Prefix for metadata')
    parser.add_argument('--output_prefix', required=True, help='S3 Prefix for output')
    parser.add_argument('--output_dir', required=True, help='Local directory to save output files')
    parser.add_argument('--num_pages_to_process', type=int, default=100, help='Number of metadata files to process from S3')
    parser.add_argument('--backend', choices=['s3', 'local'], default='s3', help='Data backend to use')
    parser.add_argument('--local_base_dir', type=str, default='data', help='Base directory for local backend')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    bucket_name = args.bucket_name
    s3_metadata_prefix = args.metadata_prefix
    data_loader = build_data_loader(
        args.backend,
        bucket_name,
        args.local_base_dir,
        checkpoint_path=PROGRESS_PATH,
    )
    # Progress checkpoint is now managed by DataLoader

    # Download the CDX parquet file from S3
    print("Reading CDX data")
    local_parquet_path = args.output_dir + "/cdx_metadata.parquet"
    os.makedirs(os.path.dirname(local_parquet_path), exist_ok=True)
    if not os.path.exists(local_parquet_path):
        data_loader.download_file(args.cdx_parquet_key, local_parquet_path)
    cdx_df = pd.read_parquet(local_parquet_path)
    cdx_df['digest'] = cdx_df['digest'].astype(str).str.replace("sha1:", "")

    print("Initializing Index")
    # Initialize the SQLite metadata index
    db_path = f'{args.output_dir}/metadata.db'
    data_loader.download_file(f'{args.output_prefix}/metadata.db', db_path)
    index = SQLiteMetadataIndex(args.output_dir)

    # Create the metadata table
    index.build_index()
    files_processed = 0
    max_files_to_process = args.num_pages_to_process * 1000
    # get the metadata files from backend
    while files_processed < max_files_to_process:
        local_metadata_path = os.path.join(args.output_dir, 'metadata_files')
        os.makedirs(local_metadata_path, exist_ok=True)
        start_time = time.time()
        print("Downloading Metadata Files from backend")
        batch_limit = min(BATCH_SIZE, max_files_to_process - files_processed)
        local_paths = data_loader.download_files(
            s3_metadata_prefix,
            local_metadata_path,
            max_keys=batch_limit,
            filter_fn=lambda key: key.endswith('.json'),
        )
        if not local_paths:
            break

        print("Adding Metadata to Index")
        rows = []
        for filepath in local_paths:
            digest = os.path.basename(os.path.dirname(filepath))
            try:
                with open(filepath, 'r') as f:
                    metadata_json = json.load(f)
                digest_val = os.path.dirname(filepath).split('/')[-1]
                rows.append({
                    'digest': digest_val,
                    'num_pages': metadata_json.get('num_pages', None)
                })
                os.remove(filepath)  # Clean up the file after reading
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

        digest_to_pagecount = pd.DataFrame(rows, columns=['digest', 'num_pages'])
        metadata_df = digest_to_pagecount.merge(cdx_df, on='digest')

        print("Building Index")
        assert args.output_prefix[-1] != '/'
        index.build_index()
        cur_batch = []
        rows_added = 0
        for _, row in metadata_df.iterrows():
            cur_batch.append({
                'crawl_url': row['url'],
                'crawl_date': row['crawl_date'],
                'pdf_name': row['digest'],
                'sub_domain': extract_subdomain(row['url']),
                'page_count': row['num_pages'],
                    's3_url': data_loader.to_uri(os.path.join('archive/2020/PDFs', f"{row['digest']}.pdf"))
            })
            if len(cur_batch) >= 1000:
                index.add_batch(cur_batch)
                rows_added += len(cur_batch)
                print(f"Added {rows_added} rows to index")
                cur_batch = []
        if len(cur_batch) > 0:
            index.add_batch(cur_batch)
            rows_added += len(cur_batch)
            print(f"Added {rows_added} rows to index")
        
        print("Uploading Index")
        data_loader.upload_file(db_path, f'{args.output_prefix}/metadata.db')

        data_loader.save_checkpoint()
        files_processed += len(local_paths)
        print("Files Processed", files_processed, "Total Time:", time.time() - start_time)
        try:
            if os.path.exists(local_metadata_path):
                shutil.rmtree(local_metadata_path)
        except Exception as e:
            print(f"Failed to remove {local_metadata_path}: {e}")
    print("Saving Index")
    index.save_index()
    
    print("Uploading Index")
    data_loader.upload_file(db_path, f'{args.output_prefix}/metadata.db')

main()
