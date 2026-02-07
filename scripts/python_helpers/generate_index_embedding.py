import os
import argparse
import time
import govscape as gs
import shutil
import json
import numpy as np
import math
from govscape.data_loader import build_data_loader

# ****************************************************************************************************
# to run this file: poetry run python s3_ec2_embedding_pipeline.py 
# ****************************************************************************************************

if __name__ == '__main__':
    # FIELDS TO SET **************************************************************************************
    parser = argparse.ArgumentParser(description="S3 EC2 Embedding Pipeline")
    parser.add_argument('--num_pages_to_process', type=int, default=100, help='Number of pages to process from S3')
    parser.add_argument('--batch_size', type=int, default=350000, help='Number of pages to process at a time')
    parser.add_argument('--bucket_name', type=str, help='S3 Bucket Name')
    parser.add_argument('--in_data_dir', type=str, help='S3 Directory for input data')
    parser.add_argument('--embedding_prefix', type=str, help='S3 Prefix for embedding files')
    parser.add_argument('--out_data_dir', type=str, help='S3 Directory for output data')
    parser.add_argument('--out_index_prefix', type=str, help='S3 Prefix for index data')
    parser.add_argument('--index_type', type=str, help='Type of index to create (e.g., "DiskANN", "FAISS")')
    parser.add_argument('--backend', choices=['s3', 'local'], default='s3', help='Data backend to use')
    parser.add_argument('--local_base_dir', type=str, default='data', help='Base directory for local backend')
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
        upload_directory_to_backend(index_directory, out_data_dir)
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
        data_loader.upload_file(perf_path, os.path.join(out_data_dir, perf_filename))
        print("finished uploading current batch")
        print("pipeline times: ", pipeline_times)

    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE):
        # result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
        # # get list of pdf file names
        # pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000
        overall_start_time = time.time()
        # Progress checkpoint is now managed by DataLoader
            
        max_files_to_process = NUM_PAGES_TO_PROCESS * 1000
        files_processed = 0
        while files_processed < max_files_to_process:

            print('*****************************************************************************************************')
            print("FILES PROCESSED: ", files_processed)
            print('*****************************************************************************************************')

            time_download = time.time()
            batch_limit = min(BATCH_SIZE, max_files_to_process - files_processed)
            local_paths = data_loader.download_files(
                in_data_dir,
                embedding_directory,
                max_keys=batch_limit,
                filter_fn=lambda key: key.endswith('.npy'),
            )
            pipeline_times['download'] += time.time() - time_download

            if not local_paths:
                break

            successful_downloads = [
                os.path.relpath(path, embedding_directory).replace("\\", "/")
                for path in local_paths
            ]
            process_embedding_files(successful_downloads)
            data_loader.save_checkpoint()

            # delete the directories except for the indices which will continue to be updated
            if os.path.exists(DATA_DIR):
                shutil.rmtree(embedding_directory)
                os.makedirs(DATA_DIR, exist_ok=True)
            
            files_processed += len(local_paths)

            # Progress checkpoint is now managed by DataLoader

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