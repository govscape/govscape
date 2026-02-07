import os
import argparse
import time
import govscape as gs
import torch
import shutil
import subprocess
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count, get_context
from govscape.data_loader import build_data_loader

# ****************************************************************************************************
# to run this file: poetry run python s3_ec2_embedding_pipeline.py 
# ****************************************************************************************************

def remove_sha1(backend, bucket, local_base_dir, sha1_keys):
    data_loader = build_data_loader(backend, bucket, local_base_dir)
    copied_properly = 0
    deleted_properly = 0
    for sha1_key in sha1_keys:
        clean_key = sha1_key.replace('sha1:', '')
        if (clean_key == sha1_key):
            continue
        correctly_copied = False
        try:
            data_loader.copy_object(sha1_key, clean_key)
            copied_properly += 1
            correctly_copied = True
        except Exception as e:
            pass
        if correctly_copied:
            try:
                data_loader.delete_object(sha1_key)
                deleted_properly += 1
            except Exception as e:
                print(f"Error deleting {sha1_key}: {e}")

    print(f"Copied {copied_properly} files to clean key.")
    print(f"Deleted {deleted_properly} files from directory.")
    return sha1_keys

if __name__ == '__main__':
    # FIELDS TO SET **************************************************************************************
    parser = argparse.ArgumentParser(description="S3 EC2 Embedding Pipeline")
    parser.add_argument('--num_pages_to_process', type=int, default=1000000, help='Number of pages to process from S3')
    parser.add_argument('--batch_size', type=int, default=1000000, help='Number of pages to process at a time')
    parser.add_argument('--bucket_name', type=str, help='S3 Bucket Name')
    parser.add_argument('--data_prefix', type=str, help='S3 Directory to be sha1 cleaned')
    parser.add_argument('--backend', choices=['s3', 'local'], default='s3', help='Data backend to use')
    parser.add_argument('--local_base_dir', type=str, default='data', help='Base directory for local backend')
    args = parser.parse_args()

    NUM_PAGES_TO_PROCESS = args.num_pages_to_process
    BATCH_SIZE = args.batch_size

    bucket_name = args.bucket_name # 'bcgl-public-bucket'
    data_prefix = args.data_prefix # 'prod-serving/'# INPUT DATA DIR IN S3 HERE

    # ****************************************************************************************************
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'prod')

    txt_directory = os.path.join(DATA_DIR, 'txt')
    index_keyword_directory = os.path.join(DATA_DIR, 'index_keyword')

    progress_path = 'sha1_clean_progress.json'  # Token to track of which pages have already been processed

    data_loader = build_data_loader(
        args.backend,
        bucket_name,
        args.local_base_dir,
        checkpoint_path=progress_path,
    )

    # gets txt files from backend
    def list_digests(num_pages=1):
        keys = []
        keys_retrieved = 0
        while True:
            result = data_loader.list_objects(data_prefix)

            key_batch = result.keys
            keys.extend(key_batch)
            keys_retrieved += 1
            if not result.is_truncated:
                return keys, True
            if keys_retrieved >= num_pages:
                break
        print(f"Listed {len(result.keys)} files from backend")
        return keys, False

    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE):
        # result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
        # # get list of pdf file names
        # pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000
        batch_size = 100
        overall_start_time = time.time()
        for i in range(0, math.ceil(NUM_PAGES_TO_PROCESS / batch_size)):
            # get the pdf files from s3
            digests, is_finished = list_digests(batch_size)
            print("Now starting with total number of PDF files: ", len(digests))

            for j in range(0, len(digests), BATCH_SIZE):
                print('*****************************************************************************************************')
                print("WE ARE ON BATCH: ", j)
                print('*****************************************************************************************************')
                batch = digests[j:j + BATCH_SIZE]
                num_workers = 64
                worker_batches = np.array_split(batch, num_workers)  # Split the batch into smaller batches for parallel downloading
                local_batch = []
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(remove_sha1, args.backend, bucket_name, args.local_base_dir, worker_batch) for worker_batch in worker_batches]
                    for future in as_completed(futures):
                        try:
                            file_name = future.result()
                            local_batch.extend(file_name)
                        except Exception as e:
                            print(f"Error downloading {future}: {e}")
                data_loader.save_checkpoint()
            if is_finished:
                break
            overall_end_time = time.time()
            print("TOTAL TIME TO RENAME IS ", (overall_end_time - overall_start_time))

    def main():
        batched_file_download(BATCH_SIZE) 

    main()