import boto3
import os
import argparse
import time
import govscape as gs
import torch
import shutil
import json
import subprocess
import numpy as np
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count, get_context
import math

# ****************************************************************************************************
# to run this file: poetry run python s3_ec2_embedding_pipeline.py 
# ****************************************************************************************************

def download_embeddings(embedding_directory, embedding_files):
    config = Config(max_pool_connections=50)
    s3 = boto3.client("s3", config=config)
    local_files = []
    for embedding_file in embedding_files:
        file_name = embedding_file.split('/')[-1]
        pdf_name = embedding_file.split('/')[-2]
        local_path = os.path.join(embedding_directory, pdf_name, file_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket_name, embedding_file, local_path)
        local_files.append(os.path.join(pdf_name, file_name))
    return local_files

if __name__ == '__main__':
    config = Config(max_pool_connections=50)
    s3 = boto3.client("s3", config=config)

    # FIELDS TO SET **************************************************************************************
    parser = argparse.ArgumentParser(description="S3 EC2 Embedding Pipeline")
    parser.add_argument('--num_pages_to_process', type=int, default=100, help='Number of pages to process from S3')
    parser.add_argument('--batch_size', type=int, default=250000, help='Number of pages to process at a time')
    parser.add_argument('--bucket_name', type=str, help='S3 Bucket Name')
    parser.add_argument('--in_data_dir', type=str, help='S3 Directory for input data')
    parser.add_argument('--embedding_prefix', type=str, help='S3 Prefix for embedding files')
    parser.add_argument('--out_data_dir', type=str, help='S3 Directory for output data')
    parser.add_argument('--out_index_prefix', type=str, help='S3 Prefix for index data')
    parser.add_argument('--index_type', type=str, help='Type of index to create (e.g., "DiskANN", "FAISS")')
    args = parser.parse_args()
    NUM_PAGES_TO_PROCESS = args.num_pages_to_process
    BATCH_SIZE = args.batch_size

    bucket_name = args.bucket_name # 'bcgl-public-bucket'
    in_data_dir = args.in_data_dir + args.embedding_prefix # 'prod-serving/'# INPUT DATA DIR IN S3 HERE
    out_data_dir = args.out_data_dir # 'prod-serving/' # OUTPUT OVERALL DATA DIR IN S3 HERE
    out_index_prefix = args.out_index_prefix # 'prod-serving/' # OUTPUT INDEX PREFIX IN S3 HERE
    index_type = args.index_type # 'FAISS' # TYPE OF INDEX TO CREATE

    # ****************************************************************************************************
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'prod')

    embedding_directory = os.path.join(DATA_DIR, args.embedding_prefix.replace('/', ''))
    index_directory = os.path.join(DATA_DIR, out_index_prefix)

    progress_path = os.path.join(PROJECT_ROOT, out_index_prefix + '_progress.json')  # Token to track of which pages have already been processed
    # ****************************************************************************************************
    # for analyzing: 
    pipeline_times = {'list' : 0, 'download' : 0, 'embedding_indexing_time' : 0, 'upload' : 0, 'pdfs_processed' : 0}  # to keep track of the time it takes for each step in the pipeline

    # gets txt files from s3
    def list_embedding_files(num_pages=1):
        is_finished = False
        embedding_files = []
        continuation_token = None
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                continuation_token = progress.get('continuation_token', None)
        
        pages_retrieved = 0
        while True:
            if continuation_token:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=in_data_dir, ContinuationToken=continuation_token)
            else:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=in_data_dir)
            print(f"Retrieved {len(result.get('Contents', []))} files from S3 Bucket: {bucket_name} Prefix: {in_data_dir}")

            contents = result.get('Contents', [])
            embedding_keys = [obj['Key'] for obj in contents if obj['Key'].endswith('.npy')]

            embedding_files.extend(embedding_keys)
            pages_retrieved += 1
            if result.get('IsTruncated'):
                continuation_token = result.get('NextContinuationToken')
            
            if pages_retrieved >= num_pages:
                break

            if not result.get('IsTruncated'):
                is_finished = True
                break
        return embedding_files, is_finished, continuation_token

    # uploads dir of files to s3
    def upload_directory_to_s3(ec2_dir, s3_dir):
        subprocess.run(f"/home/ubuntu/.local/bin/s5cmd --log error cp {ec2_dir} s3://{bucket_name}/{s3_dir}/".split())

    # processing the pdfs: running through embedding pipeline and uploading to s3
    def process_embedding_files(embedding_files):
        time_index_start = time.time()
        index = gs.FAISSIndex(index_directory)
        index.load_index()
        names = []
        pages = []
        embeddings = []
        for embedding_file in embedding_files:
            embedding_file_path = os.path.join(embedding_directory, embedding_file)
            if not os.path.exists(embedding_file_path):
                print(f"File {embedding_file_path} does not exist. Skipping.")
                continue
            names.append(os.path.basename(os.path.dirname(embedding_file_path)))
            pages.append(embedding_file_path.replace(".npy", "").rpartition('_')[2])
            embeddings.append(np.load(embedding_file_path))
        embeddings = np.asarray(embeddings)
        index.add_batch(embeddings, names, pages)
        index.save_index()

        pipeline_times['embedding_indexing_time'] += time.time() - time_index_start

        time1 = time.time()
        # UPLOADING Indexes TO S3 HERE
        upload_directory_to_s3(index_directory, out_data_dir)
        print("finished uploading index")
        time2 = time.time()

        pipeline_times['upload'] += time2-time1
        pipeline_times['pdfs_processed'] += len(embedding_files)
        
        # Write pipeline_times to a JSON file
        perf_filename = f"{out_index_prefix}_performance.json"
        perf_path = os.path.join(DATA_DIR, perf_filename)
        with open(perf_path, "w") as f:
            json.dump(pipeline_times, f, indent=2)

        # Upload the performance JSON to S3
        s3.upload_file(perf_path, bucket_name, os.path.join(out_data_dir, perf_filename))
        print("finished uploading current batch")
        print("pipeline times: ", pipeline_times)


    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE):
        # result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
        # # get list of pdf file names
        # pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000

        overall_start_time = time.time()

        # get the pdf files from s3
        pages_processed = 0
        pages_per_batch = math.floor(BATCH_SIZE / 1000)
        while pages_processed < NUM_PAGES_TO_PROCESS:

            print('*****************************************************************************************************')
            print("WE ARE ON BATCH: ", pages_processed * 1000)
            print('*****************************************************************************************************')

            time_list = time.time()
            embedding_files, finished, continuation_token = list_embedding_files(pages_per_batch)
            pipeline_times['list'] += time.time() - time_list
            successful_downloads = []
            time_download = time.time()
            worker_batches = np.array_split(embedding_files, 64)  # Split the batch into 64 smaller batches for parallel downloading

            with ProcessPoolExecutor(max_workers=64) as executor:
                futures = [executor.submit(download_embeddings, embedding_directory, worker_batch) for worker_batch in worker_batches]
                for future in as_completed(futures):
                    try:
                        file_names = future.result()
                        successful_downloads.extend(file_names)
                    except Exception as e:
                        print(f"Error downloading {futures[future]}: {e}")
            pipeline_times['download'] += time.time() - time_download

            process_embedding_files(successful_downloads)

            # delete the directories except for the indices which will continue to be updated
            if os.path.exists(DATA_DIR):
                shutil.rmtree(embedding_directory)
                os.makedirs(DATA_DIR, exist_ok=True)
            
            # If we didn't get a full set of embedding files,
            if finished:
                break

            pages_processed += pages_per_batch

            # Save continuation token for next run
            try:
                with open(progress_path, 'w') as f:
                    json.dump({'continuation_token': continuation_token}, f)
            except Exception as e:
                print(f"Error saving continuation token: {e}")
        
        # After all batches are processed, clean up the directories
        if os.path.exists(embedding_directory):
            shutil.rmtree(embedding_directory)
        if os.path.exists(index_directory):
            shutil.rmtree(index_directory)

        overall_end_time = time.time()
        print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))

    def main():
        batched_file_download(BATCH_SIZE) 

    main()