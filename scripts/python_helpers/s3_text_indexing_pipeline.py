import boto3
import os
import argparse
import time
import govscape as gs
import torch
import shutil
import json
import subprocess
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count, get_context

# ****************************************************************************************************
# to run this file: poetry run python s3_ec2_embedding_pipeline.py 
# ****************************************************************************************************

if __name__ == '__main__':
    config = Config(max_pool_connections=50)
    s3 = boto3.client("s3", config=config)

    # FIELDS TO SET **************************************************************************************
    parser = argparse.ArgumentParser(description="S3 EC2 Embedding Pipeline")
    parser.add_argument('--num_pages_to_process', type=int, default=100, help='Number of pages to process from S3')
    parser.add_argument('--batch_size', type=int, default=10000, help='Number of pages to process at a time')
    parser.add_argument('--bucket_name', type=str, help='S3 Bucket Name')
    parser.add_argument('--in_data_dir', type=str, help='S3 Directory for input data')
    parser.add_argument('--out_data_dir', type=str, help='S3 Directory for output data')
    args = parser.parse_args()

    NUM_PAGES_TO_PROCESS = args.num_pages_to_process
    BATCH_SIZE = args.batch_size

    bucket_name = args.bucket_name # 'bcgl-public-bucket'
    in_data_dir = args.in_data_dir # 'prod-serving/'# INPUT DATA DIR IN S3 HERE 
    out_data_dir = args.out_data_dir # 'prod-serving/' # OUTPUT OVERALL DATA DIR IN S3 HERE 

    # ****************************************************************************************************
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'prod')

    txt_directory = os.path.join(DATA_DIR, 'txt')
    index_keyword_directory = os.path.join(DATA_DIR, 'index_keyword')

    progress_path = 'text_index_progress.json'  # Token to track of which pages have already been processed

    # ****************************************************************************************************
    # for analyzing: 
    pipeline_times = {'list' : 0, 'download' : 0, 'keyword_indexing_time' : 0, 'upload' : 0}  # to keep track of the time it takes for each step in the pipeline

    # gets txt files from s3
    def list_txt_files(num_pages=1):
        txt_files = []
        continuation_token = None
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                continuation_token = progress.get('continuation_token', None)
        pages_retrieved = 0
        while True:
            if continuation_token:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=in_data_dir + "/txt", ContinuationToken=continuation_token)
            else:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=in_data_dir + "/txt")
            print(f"Retrieved {len(result.get('Contents', []))} files from S3")
            
            contents = result.get('Contents', [])
            txt_keys = [obj['Key'] for obj in contents if obj['Key'].endswith('.txt')]

            txt_files.extend(txt_keys)
            pages_retrieved += 1
            if result.get('IsTruncated'):
                continuation_token = result.get('NextContinuationToken')
                with open(progress_path, 'w') as f:
                    json.dump({'continuation_token': continuation_token}, f)
            
            if pages_retrieved >= num_pages or not result.get('IsTruncated'):
                break

        return txt_files

    # uploads dir of files to s3
    def upload_directory_to_s3(ec2_dir, s3_dir):
        subprocess.run(f"s5cmd --log error cp {ec2_dir} s3://{bucket_name}/{s3_dir}/".split())

    # processing the pdfs: running through embedding pipeline and uploading to s3
    def process_txt_files(txt_files):
        start_time = time.time()

        time_index_start = time.time()
        index = gs.WhooshIndex(index_keyword_directory)
        index.load_index()
        names = []
        pages = []
        txts = []
        for txt_file in txt_files:
            txt_file_path = os.path.join(txt_directory, txt_file)
            if not os.path.exists(txt_file_path):
                print(f"File {txt_file_path} does not exist. Skipping.")
                continue
            names.append(txt_file_path.rpartition('/')[0])
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
        upload_directory_to_s3(index_keyword_directory, out_data_dir)
        print("finished uploading keyword index")
        time2 = time.time()

        pipeline_times['upload'] += time2-time1
        print("finished uploading current batch")
        print("pipeline times: ", pipeline_times)


    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE):
        # result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
        # # get list of pdf file names
        # pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000

        overall_start_time = time.time()

        # get the pdf files from s3
        time_list = time.time()
        txt_files = list_txt_files(NUM_PAGES_TO_PROCESS)
        pipeline_times['list'] = time.time() - time_list

        print("Now starting with total number of PDF files: ", len(txt_files))

        for i in range(0, len(txt_files), BATCH_SIZE):
            print('*****************************************************************************************************')
            print("WE ARE ON BATCH: ", i)
            print('*****************************************************************************************************')
            batch = txt_files[i:i + BATCH_SIZE] 
            local_batch = []
            time_download = time.time()
            def download_txt(txt_file):
                file_name = txt_file.split('/')[-1]
                pdf_name = txt_file.split('/')[-2]
                local_path = os.path.join(txt_directory, pdf_name, file_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket_name, txt_file, local_path)
                return os.path.join(pdf_name, file_name)

            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(download_txt, pdf): pdf for pdf in batch}
                for future in as_completed(futures):
                    try:
                        file_name = future.result()
                        local_batch.append(file_name)
                    except Exception as e:
                        print(f"Error downloading {futures[future]}: {e}")
            pipeline_times['download'] += time.time() - time_download

            process_txt_files(local_batch)

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