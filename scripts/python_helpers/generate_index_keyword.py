import os
import argparse
import time
import govscape as gs
import torch
import shutil
import json
import math
import numpy as np
from multiprocessing import Pool, cpu_count, get_context
from govscape.data_loader import build_data_loader

# ****************************************************************************************************
# to run this file: poetry run python s3_ec2_embedding_pipeline.py 
# ****************************************************************************************************

if __name__ == '__main__':
    # FIELDS TO SET **************************************************************************************
    parser = argparse.ArgumentParser(description="S3 EC2 Embedding Pipeline")
    parser.add_argument('--num_pages_to_process', type=int, default=100, help='Number of pages to process from S3')
    parser.add_argument('--batch_size', type=int, default=100000, help='Number of pages to process at a time')
    parser.add_argument('--bucket_name', type=str, help='S3 Bucket Name')
    parser.add_argument('--in_data_dir', type=str, help='S3 Directory for input data')
    parser.add_argument('--out_data_dir', type=str, help='S3 Directory for output data')
    parser.add_argument('--keyword_index_type', type=str, default='LanceDB', help='Type of keyword index to use: LanceDB, SQLite or Whoosh')
    parser.add_argument('--backend', choices=['s3', 'local'], default='s3', help='Data backend to use')
    parser.add_argument('--local_base_dir', type=str, default='data', help='Base directory for local backend')
    args = parser.parse_args()

    NUM_PAGES_TO_PROCESS = args.num_pages_to_process
    BATCH_SIZE = args.batch_size
    index_type = args.keyword_index_type # 'LanceDB', 'SQLite' or 'Whoosh'
    bucket_name = args.bucket_name # 'bcgl-public-bucket'
    in_data_dir = args.in_data_dir # 'prod-serving/'# INPUT DATA DIR IN S3 HERE 
    out_data_dir = args.out_data_dir # 'prod-serving/' # OUTPUT OVERALL DATA DIR IN S3 HERE 

    # ****************************************************************************************************
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'prod')

    txt_directory = os.path.join(DATA_DIR, 'txt')
    index_keyword_directory = os.path.join(DATA_DIR, 'index_keyword')

    progress_filename = 'text_index_progress.json'  # Token to track of which pages have already been processed
    progress_path = os.path.join(DATA_DIR, progress_filename)

    perf_filename = "text_indexing_performance.json"
    perf_path = os.path.join(DATA_DIR, perf_filename)
    
    # ****************************************************************************************************
    # for analyzing:     
    pipeline_times = {'list' : 0, 'download' : 0, 'keyword_indexing_time' : 0, 'upload' : 0, 'pdfs_processed' : 0}  # to keep track of the time it takes for each step in the pipeline

    data_loader = build_data_loader(
        args.backend,
        bucket_name,
        args.local_base_dir,
        checkpoint_path=progress_path,
    )

    # uploads dir of files to backend
    def upload_directory_to_backend(local_dir, remote_dir):
        data_loader.upload_directory(local_dir, remote_dir)

    # processing the pdfs: running through embedding pipeline and uploading to s3
    def process_txt_files(txt_files):
        start_time = time.time()

        time_index_start = time.time()
        if index_type == "LanceDB":
            index = gs.LanceDBKeywordIndex(index_keyword_directory)
        elif index_type == "SQLite":
            index = gs.SQLiteKeywordIndex(index_keyword_directory)
        elif index_type == "Whoosh":
            index = gs.WhooshKeywordIndex(index_keyword_directory)
        else:
            raise ValueError("index_type must be either 'LanceDB', 'SQLite', or 'Whoosh'")
        index.load_index()
        names = []
        pages = []
        txts = []
        for txt_file in txt_files:
            txt_file_path = os.path.join(txt_directory, txt_file)
            if not os.path.exists(txt_file_path):
                print(f"File {txt_file_path} does not exist. Skipping.")
                continue
            names.append(os.path.basename(os.path.dirname(txt_file_path)))
            pages.append(txt_file_path.replace(".txt", "").rpartition('_')[2])
            txt = None 
            with open(txt_file_path) as f:
                txt = f.read()
            txts.append(txt)
        index.add_batch(txts, names, pages)
        index.save_index()

        end_time = time.time()
        duration = end_time - start_time
        if duration > 0:
            throughput = len(txt_files) / duration
        else:
            throughput = 0
        pipeline_times['keyword_indexing_time'] += time.time() - time_index_start 
        
        time1 = time.time()
        # UPLOADING Indexes TO S3 HERE 
        upload_directory_to_backend(index_keyword_directory, out_data_dir)
        print("finished uploading keyword index")
        time2 = time.time()

        pipeline_times['upload'] += time2-time1
        pipeline_times['pdfs_processed'] += len(txt_files)
        



    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE):
        try:
            data_loader.download_file(os.path.join(out_data_dir, perf_filename), perf_path)
            with open(perf_path, "r") as f:
                existing_pipeline_times = json.load(f)
                for key in pipeline_times:
                    pipeline_times[key] = existing_pipeline_times.get(key, 0)
        except Exception:
            print("No existing performance file found. Starting fresh.")
        overall_start_time = time.time()
        files_processed = 0
        max_files_to_process = NUM_PAGES_TO_PROCESS * 1000
        while files_processed < max_files_to_process:

            print('*****************************************************************************************************')
            print("FILES PROCESSED: ", files_processed)
            print('*****************************************************************************************************')


            time_download = time.time()
            batch_limit = min(BATCH_SIZE, max_files_to_process - files_processed)
            local_paths = data_loader.download_files(
                in_data_dir + "/txt",
                txt_directory,
                max_keys=batch_limit,
                filter_fn=lambda key: key.endswith('.txt'),
            )
            pipeline_times['download'] += time.time() - time_download

            if not local_paths:
                break

            local_batch = [
                os.path.relpath(path, txt_directory).replace("\\", "/")
                for path in local_paths
            ]
            process_txt_files(local_batch)

            # Write continuation token to progress file
            data_loader.save_checkpoint()
                
            # Write pipeline_times to a JSON file
            with open(perf_path, "w") as f:
                json.dump(pipeline_times, f, indent=2)

            # Upload the performance json to backend
            data_loader.upload_file(perf_path, os.path.join(out_data_dir, perf_filename))
            print("finished uploading current batch")
            print("pipeline times: ", pipeline_times)
            files_processed += len(local_paths)

            # delete the directories except for the indices which will continue to be updated
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR + "/txt")
                os.makedirs(DATA_DIR, exist_ok=True)

        
        # After all batches are processed, clean up the directories
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR, exist_ok=True)

        overall_end_time = time.time()
        print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))

    def main():
        batched_file_download(BATCH_SIZE) 

    main()