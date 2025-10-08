import subprocess
from govscape.indexing import SQLiteMetadataIndex
import json
import argparse
import pandas as pd
from urllib.parse import urlparse
import time
import boto3
import os
import multiprocessing

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
NUM_PAGES_TO_PROCESS = 1
BATCH_SIZE = 10000
# gets metadata files from s3
def list_metadata_files(s3, bucket_name, s3_metadata_prefix, num_pages=1):
    metadata_files = []
    continuation_token = None
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            progress = json.load(f)
            continuation_token = progress.get('continuation_token', None)
    pages_retrieved = 0
    while True:
        if continuation_token:
            result = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_metadata_prefix, ContinuationToken=continuation_token)
        else:
            result = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_metadata_prefix)
        print(f"Retrieved {len(result.get('Contents', []))} files from S3")
        
        contents = result.get('Contents', [])
        metadata_keys = [obj['Key'] for obj in contents if obj['Key'].endswith('.json')]
        metadata_files.extend(metadata_keys)
        pages_retrieved += 1
        if result.get('IsTruncated'):
            continuation_token = result.get('NextContinuationToken')
            finished = False

        if pages_retrieved >= num_pages or not result.get('IsTruncated'):
            finished = True
            break
    return continuation_token, finished, metadata_files

def download_files_from_s3(bucket_name, s3_keys, local_path):
    s3 = boto3.client('s3')
    for s3_key in s3_keys:
        digest = os.path.dirname(s3_key).split('/')[-1]
        local_file_path = os.path.join(local_path, digest, "metadata.json")
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        s3.download_file(bucket_name, s3_key, local_file_path)

def main():    
    parser = argparse.ArgumentParser(description='Process CDX files from S3.')
    parser.add_argument('--bucket_name', required=True, help='S3 bucket name')
    parser.add_argument('--cdx_parquet_key', required=True, help='S3 Key for CDX parquet file')
    parser.add_argument('--metadata_prefix', required=True, help='S3 Prefix for metadata')
    parser.add_argument('--output_prefix', required=True, help='S3 Prefix for output')
    parser.add_argument('--output_dir', required=True, help='Local directory to save output files')
    parser.add_argument('--num_pages_to_process', type=int, default=100, help='Number of metadata files to process from S3')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    s3 = boto3.client('s3')
    bucket_name = args.bucket_name
    s3_metadata_prefix = args.metadata_prefix
    # get the pdf files from s3
    continuation_token, is_finished, metadata_files = list_metadata_files(s3, bucket_name, s3_metadata_prefix, args.num_pages_to_process)

    # Download the CDX parquet file from S3
    print("Reading CDX data")
    s3 = boto3.client('s3')
    local_parquet_path = args.output_dir + "/cdx_metadata.parquet"
    os.makedirs(os.path.dirname(local_parquet_path), exist_ok=True)
    if not os.path.exists(local_parquet_path):
        s3.download_file(bucket_name, args.cdx_parquet_key, local_parquet_path)
    cdx_df = pd.read_parquet(local_parquet_path)
    cdx_df['digest'] = cdx_df['digest'].astype(str).str.replace("sha1:", "")

    metadata_file_batches = [metadata_files[i:i + BATCH_SIZE] for i in range(0, len(metadata_files), BATCH_SIZE)]
    local_metadata_path = os.path.join(args.output_dir, 'metadata')
    os.makedirs(local_metadata_path, exist_ok=True)
    start_time = time.time()
    for i, metadata_file_batch in enumerate(metadata_file_batches):
        download_batches = [metadata_file_batch[i:i + 1000] for i in range(0, len(metadata_file_batch), 1000)]
        with multiprocessing.Pool(processes=64) as pool:
            pool.starmap(download_files_from_s3, [(bucket_name, download_batch, local_metadata_path) for download_batch in download_batches])

        rows = []
        for metadata_file in metadata_file_batch:
            digest = os.path.dirname(metadata_file).split('/')[-1]
            filepath = os.path.join(local_metadata_path, digest, "metadata.json")
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


        # Initialize the SQLite metadata index
        db_path = f'{args.output_dir}/metadata.db'
        if os.path.exists(db_path):
            os.remove(db_path)
        index = SQLiteMetadataIndex(args.output_dir)

        # Create the metadata table
        index.build_index()

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
                's3_url': f"https://{args.bucket_name}.s3.amazonaws.com/archive/2020/PDFs/{row['digest']}.pdf"
            })
            if len(cur_batch) >= 1000:
                index.add_batch(cur_batch)
                rows_added += len(cur_batch)
                print(f"Added {rows_added} rows to index")
                cur_batch = []
        
        print("Uploading Index")
        s3 = boto3.client('s3')
        s3.upload_file(db_path, 'bcgl-public-bucket', f'{args.output_prefix}/metadata.db')

        with open(PROGRESS_PATH, 'w') as f:
            json.dump({'continuation_token': continuation_token}, f)
        print(f'Processed Batch {i+1} out of {len(metadata_file_batches)} in {time.time() - start_time:.2f} seconds')

    print("Saving Index")
    index.save_index()
    
    print("Uploading Index")
    s3 = boto3.client('s3')
    s3.upload_file(db_path, 'bcgl-public-bucket', f'{args.output_prefix}/metadata.db')

main()
