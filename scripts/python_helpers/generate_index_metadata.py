from govscape.indexing import SQLiteMetadataIndex
import json
import argparse
import pandas as pd
from urllib.parse import urlparse
import time
import os
import shutil
from govscape.data_loader import RemoteDirectoryIterator, build_data_loader

def extract_subdomain(url):
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return None
    parts = hostname.split('.')
    if len(parts) >= 2:
        return '.'.join(parts[-2:])
    return hostname

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
    BUCKET_NAME = args.bucket_name
    NUM_PAGES_TO_PROCESS = args.num_pages_to_process

    #*********************************************************************************************
    LOCAL_DATA_DIR = args.output_dir
    REMOTE_METADATA_DIR = args.metadata_prefix
    OUT_DATA_DIR = args.output_prefix
    LOCAL_METADATA_DIR = os.path.join(LOCAL_DATA_DIR, 'metadata_files')
    LOCAL_PARQUET_PATH = os.path.join(LOCAL_DATA_DIR, 'cdx_metadata.parquet')
    LOCAL_INDEX_PATH = os.path.join(LOCAL_DATA_DIR, 'metadata.db')
    REMOTE_INDEX_PATH = f'{OUT_DATA_DIR}/metadata.db'
    REMOTE_CHECKPOINT_PATH = f'{OUT_DATA_DIR}/Checkpoints/checkpoint_metadata.json'
    LOCAL_CHECKPOINT_PATH = os.path.join(LOCAL_DATA_DIR, 'Checkpoints', 'checkpoint_metadata.json')

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(LOCAL_METADATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOCAL_CHECKPOINT_PATH), exist_ok=True)

    #*********************************************************************************************
    data_loader = build_data_loader(
        args.backend,
        BUCKET_NAME,
        local_base_dir=args.local_base_dir,
    )

    remote_iter = RemoteDirectoryIterator(
        data_loader,
        REMOTE_METADATA_DIR,
        remote_checkpoint_path=REMOTE_CHECKPOINT_PATH,
        local_checkpoint_path=LOCAL_CHECKPOINT_PATH,
        local_dir=LOCAL_METADATA_DIR,
    )

    # Download the CDX parquet file from S3
    print("Reading CDX data")
    os.makedirs(os.path.dirname(LOCAL_PARQUET_PATH), exist_ok=True)
    if not os.path.exists(LOCAL_PARQUET_PATH):
        data_loader.download_file(args.cdx_parquet_key, LOCAL_PARQUET_PATH)
    cdx_df = pd.read_parquet(LOCAL_PARQUET_PATH)
    cdx_df['digest'] = cdx_df['digest'].astype(str).str.replace("sha1:", "")

    print("Initializing Index")

    # Initialize the SQLite metadata index
    try:
        data_loader.download_file(REMOTE_INDEX_PATH, LOCAL_INDEX_PATH)
    except Exception as e:
        print(f"No Existing Index File Found: {e}")
    index = SQLiteMetadataIndex(LOCAL_DATA_DIR)

    # Create the metadata table
    index.build_index()
    files_processed = 0
    max_files_to_process = NUM_PAGES_TO_PROCESS * 1000
    # get the metadata files from backend
    while files_processed < max_files_to_process:
        os.makedirs(LOCAL_METADATA_DIR, exist_ok=True)
        start_time = time.time()
        print("Downloading Metadata Files from backend")
        batch_limit = min(BATCH_SIZE, max_files_to_process - files_processed)
        local_paths = remote_iter.download_batch(
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
        data_loader.upload_file(LOCAL_INDEX_PATH, REMOTE_INDEX_PATH)

        remote_iter.save_checkpoint()
        files_processed += len(local_paths)
        print("Files Processed", files_processed, "Total Time:", time.time() - start_time)
        try:
            if os.path.exists(LOCAL_METADATA_DIR):
                shutil.rmtree(LOCAL_METADATA_DIR)
        except Exception as e:
            print(f"Failed to remove {LOCAL_METADATA_DIR}: {e}")
    print("Saving Index")
    index.save_index()
    
    print("Uploading Index")
    data_loader.upload_file(LOCAL_INDEX_PATH, REMOTE_INDEX_PATH)

main()
