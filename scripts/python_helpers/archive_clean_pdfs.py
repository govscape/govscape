import boto3
import os
import argparse
import time
import json
import math
import numpy as np
from botocore.config import Config
from concurrent.futures import ProcessPoolExecutor, as_completed

# ****************************************************************************************************
# to run this file: poetry run python s3_ec2_embedding_pipeline.py 
# ****************************************************************************************************

def copy_to_clean(bucket, dirty_prefix, clean_prefix, digests):
    config = Config(max_pool_connections=50)
    s3 = boto3.client("s3", config=config)
    copied_properly = 0
    for digest in digests:
        try:
            clean_digest = digest.replace('sha1:', '')
            clean_file = os.path.join(clean_prefix, clean_digest + '.pdf')
            dirty_file = os.path.join(dirty_prefix, digest + '.pdf')
            response = s3.copy_object(
                Bucket=bucket,
                CopySource=f'/{bucket}/{dirty_file}',
                Key=clean_file
            )
            copied_properly += 1
        except Exception as e:
            pass

    print(f"Copied {copied_properly} files to clean directory.")
    return digests

if __name__ == '__main__':
    config = Config(max_pool_connections=50)
    s3 = boto3.client("s3", config=config)

    # FIELDS TO SET **************************************************************************************
    parser = argparse.ArgumentParser(description="S3 EC2 Embedding Pipeline")
    parser.add_argument('--num_pages_to_process', type=int, default=1000000, help='Number of pages to process from S3')
    parser.add_argument('--batch_size', type=int, default=1000000, help='Number of pages to process at a time')
    parser.add_argument('--bucket_name', type=str, help='S3 Bucket Name')
    parser.add_argument('--metadata_prefix', type=str, help='S3 Directory for metadata data')
    parser.add_argument('--clean_data_prefix', type=str, help='S3 Directory for input data')
    parser.add_argument('--dirty_data_prefix', type=str, help='S3 Directory for output data')
    args = parser.parse_args()

    NUM_PAGES_TO_PROCESS = args.num_pages_to_process
    BATCH_SIZE = args.batch_size

    bucket_name = args.bucket_name # 'bcgl-public-bucket'
    metadata_prefix = args.metadata_prefix # 'prod-serving/'# INPUT DATA DIR IN S3 HERE
    clean_data_prefix = args.clean_data_prefix # 'prod-serving/'# INPUT DATA DIR IN S3 HERE
    dirty_data_prefix = args.dirty_data_prefix # 'prod-serving/' # OUTPUT OVERALL DATA DIR IN S3 HERE

    # ****************************************************************************************************
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'prod')

    txt_directory = os.path.join(DATA_DIR, 'txt')
    index_keyword_directory = os.path.join(DATA_DIR, 'index_keyword')

    progress_path = 'clean_copy_progress.json'  # Token to track of which pages have already been processed

    # gets txt files from s3
    def list_digests(num_pages=1):
        digests = []
        continuation_token = None
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                continuation_token = progress.get('continuation_token', None)
        pdfs_retrieved = 0
        while True:
            if continuation_token:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=metadata_prefix, ContinuationToken=continuation_token)
            else:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=metadata_prefix)
            
            contents = result.get('Contents', [])
            digest_batch = [os.path.split(os.path.split(obj['Key'])[0])[1] for obj in contents if obj['Key'].endswith('.json')]
            digests.extend(digest_batch)
            pdfs_retrieved += 1
            if result.get('IsTruncated'):
                continuation_token = result.get('NextContinuationToken')
#                with open(progress_path, 'w') as f:
#                    json.dump({'continuation_token': continuation_token}, f)
            print(contents)
            if not result.get('IsTruncated'):
                return digests, False
            if pdfs_retrieved >= num_pages or not result.get('IsTruncated'):
                break
        print(f"Listed {len(result.get('Contents', []))} files from S3")
        return digests, True

    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE):
        # result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
        # # get list of pdf file names
        # pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000

        overall_start_time = time.time()
        for i in range(1, math.ceil(NUM_PAGES_TO_PROCESS / 1000)):
            # get the pdf files from s3
            digests, is_finished = list_digests(1000)
            print("Now starting with total number of PDF files: ", len(digests))

            for j in range(0, len(digests), BATCH_SIZE):
                print('*****************************************************************************************************')
                print("WE ARE ON BATCH: ", j)
                print('*****************************************************************************************************')
                batch = digests[j:j + BATCH_SIZE]
                num_workers = 512
                worker_batches = np.array_split(batch, num_workers)  # Split the batch into smaller batches for parallel downloading
                local_batch = []
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(copy_to_clean, bucket_name, dirty_data_prefix, clean_data_prefix, worker_batch) for worker_batch in worker_batches]
                    for future in as_completed(futures):
                        try:
                            file_name = future.result()
                            local_batch.extend(file_name)
                        except Exception as e:
                            print(f"Error downloading {futures[future]}: {e}")
            if is_finished:
                break
            overall_end_time = time.time()
            print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))

    def main():
        batched_file_download(BATCH_SIZE) 

    main()