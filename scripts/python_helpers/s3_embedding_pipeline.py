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
from botocore.exceptions import ClientError
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
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of pdfs to process at a time')
    parser.add_argument('--bucket_name', type=str, help='S3 Bucket Name')
    parser.add_argument('--pdf_dir', type=str, help='S3 Directory containing PDFs')
    parser.add_argument('--data_dir', type=str, help='S3 Directory for output data')
    parser.add_argument('--model_type', type=str, help='The model type to use for embedding', default='ST')
    parser.add_argument('--num_servers', type=int, help='The number of servers to use for embedding', default=1)
    parser.add_argument('--server_id', type=int, help='The ID of the current server', default=0)
    args = parser.parse_args()

    NUM_PAGES_TO_PROCESS = args.num_pages_to_process
    BATCH_SIZE = args.batch_size

    # s3://bcgl-public-bucket/2008_EOT_PDFs/PDFs/
    bucket_name = args.bucket_name # 'bcgl-public-bucket'
    pdfs_dir = args.pdf_dir # 'archive/2020/PDFs/'# INPUT DATA DIR IN S3 HERE 
    data_dir_s3 = args.data_dir # 'prod-serving/' # OUTPUT OVERALL DATA DIR IN S3 HERE 

    # ****************************************************************************************************
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'prod')

    pdf_directory = os.path.join(DATA_DIR, 'PDFs')
    txt_directory = os.path.join(DATA_DIR, 'txt')
    image_directory = os.path.join(DATA_DIR, 'img')
    img_extracted_dir = os.path.join(DATA_DIR, 'img_extracted')
    embeddings_directory = os.path.join(DATA_DIR, 'embeddings')
    embeddings_img_pg_directory = os.path.join(DATA_DIR, 'embeddings_img_pg')
    embeddings_img_extracted_directory = os.path.join(DATA_DIR, 'embeddings_img_extracted')
    metadata_dir = os.path.join(DATA_DIR, 'metadata')
    index_directory = os.path.join(DATA_DIR, 'index')
    index_img_directory = os.path.join(DATA_DIR, 'index_img')

    if args.model_type == "ST":
        text_model = gs.ST_TextEmbeddingModel()
    elif args.model_type == "BGE":
        text_model = gs.BGE_TextEmbeddingModel()
    else:
        raise ValueError("Unsupported model type")

    devices = []
    for i in range(torch.cuda.device_count()):
        devices.append("cuda:" + str(i))
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

    model_pool = text_model.model.start_multi_process_pool(target_devices=devices)
    processor = gs.PDFsToEmbeddings(pdf_directory, DATA_DIR, text_model, model_pool)

    progress_path = 'progress.json'  # Token to track of which pages have already been processed

    # ****************************************************************************************************
    # for analyzing: 
    pipeline_times = {'list' : 0, 'download' : 0, 'pdf_to_txt_img': 0, 'text_embed_time': 0, 'img_embed_time': 0, 'metadata_time': 0, 'upload' : 0, 'pdfs_processed' : 0}  # to keep track of the time it takes for each step in the pipeline

    # gets pdfs from s3
    def list_pdfs(num_pages=1):
        pdf_files = []
        continuation_token = None
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                continuation_token = progress.get('continuation_token', None)
        pages_retrieved = 0
        while True:
            if continuation_token:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir, ContinuationToken=continuation_token)
            else:
                result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
            
            contents = result.get('Contents', [])
            pdf_keys = [obj['Key'] for obj in contents if obj['Key'].endswith('.pdf')]
            pdf_keys = [key for key in pdf_keys if (hash(key) % args.num_servers) == args.server_id]

            pdf_files.extend(pdf_keys)
            pages_retrieved += 1
            if result.get('IsTruncated'):
                continuation_token = result.get('NextContinuationToken')
                with open(progress_path, 'w') as f:
                    json.dump({'continuation_token': continuation_token}, f)
            
            if pages_retrieved >= num_pages or not result.get('IsTruncated'):
                break

        return pdf_files

    # uploads dir of files to s3
    def upload_directory_to_s3(ec2_dir, s3_dir):
        subprocess.run(f"/home/ubuntu/.local/bin/s5cmd --log error cp {ec2_dir} s3://{bucket_name}/{s3_dir}".split())

    # processing the pdfs: running through embedding pipeline and uploading to s3
    def process_pdfs(pdf_files, processor):

        # PROCESS PDFS HERE 
        pdf_to_txt_img, text_embed_time, img_embed_time, metadata_time = processor.pdfs_to_embeddings(pdf_files=pdf_files)
        pipeline_times['pdf_to_txt_img'] += pdf_to_txt_img
        pipeline_times['text_embed_time'] += text_embed_time
        pipeline_times['img_embed_time'] += img_embed_time 
        pipeline_times['metadata_time'] += metadata_time

        time1 = time.time()
        # UPLOADING EMBEDDINGS, TXTS, IMAGES TO S3 HERE 
        upload_directory_to_s3(txt_directory, data_dir_s3)
        print("finished uploading txt")
        upload_directory_to_s3(image_directory, data_dir_s3)
        print("finished uploading img")
        upload_directory_to_s3(img_extracted_dir, data_dir_s3)
        print("finished uploading img extracted")
        upload_directory_to_s3(embeddings_directory, data_dir_s3)
        print("finished uploading embeddings")
        upload_directory_to_s3(embeddings_img_pg_directory, data_dir_s3)
        print("finished uploading embed img pg")
        upload_directory_to_s3(embeddings_img_extracted_directory, data_dir_s3)
        print("finished uploading embed img extracted")
        upload_directory_to_s3(metadata_dir, data_dir_s3)
        print("finished uploading metadata")
        upload_directory_to_s3(index_directory, data_dir_s3)
        print("finished uploading embedding index")
        upload_directory_to_s3(index_img_directory, data_dir_s3)
        print("finished uploading image embedding index")

        time2 = time.time()

        pipeline_times['upload'] += time2-time1
        pipeline_times['pdfs_processed'] += len(pdf_files)

        # Write pipeline_times to a JSON file
        perf_filename = f"performance_{args.server_id}.json"
        perf_path = os.path.join(DATA_DIR, perf_filename)
        with open(perf_path, "w") as f:
            json.dump(pipeline_times, f, indent=2)

        # Upload the performance JSON to S3
        s3.upload_file(perf_path, bucket_name, os.path.join(data_dir_s3, perf_filename))
        
        print("finished uploading current batch")
        print("pipeline times: ", pipeline_times)


    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE, processor):
        # result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
        # # get list of pdf file names
        # pdf_files = [obj['Key'] for obj in result.get('Contents', []) if obj['Key'].endswith('.pdf')]  # note this only returns 1000

        overall_start_time = time.time()

        # get the pdf files from s3
        time_list = time.time()
        pdf_files = list_pdfs(NUM_PAGES_TO_PROCESS)
        pipeline_times['list'] = time.time() - time_list

        print("Now starting with total number of PDF files: ", len(pdf_files))

        for i in range(0, len(pdf_files), BATCH_SIZE):
            print('*****************************************************************************************************')
            print("WE ARE ON BATCH: ", i)
            print('*****************************************************************************************************')
            batch = pdf_files[i:i + BATCH_SIZE] 
            local_batch = []
            time_download = time.time()
            def download_pdf(pdf):
                file_name = os.path.basename(pdf)
                local_path = os.path.join(pdf_directory, file_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket_name, pdf, local_path)
                return file_name

            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(download_pdf, pdf): pdf for pdf in batch}
                for future in as_completed(futures):
                    try:
                        file_name = future.result()
                        local_batch.append(file_name)
                    except Exception as e:
                        print(f"Error downloading {futures[future]}: {e}")
            pipeline_times['download'] += time.time() - time_download

            process_pdfs(local_batch, processor)

            # delete the directories except for the indices which will continue to be updated
            if os.path.exists(DATA_DIR):
                shutil.rmtree(DATA_DIR + "/embeddings")
                shutil.rmtree(DATA_DIR + "/embeddings_img_pg")
                shutil.rmtree(DATA_DIR + "/txt")
                shutil.rmtree(DATA_DIR + "/img")
                shutil.rmtree(DATA_DIR + "/metadata")
                os.makedirs(DATA_DIR, exist_ok=True)

            if os.path.exists(pdf_directory):
                shutil.rmtree(pdf_directory)
                os.makedirs(pdf_directory, exist_ok=True)
        
        # After all batches are processed, clean up the directories
        if os.path.exists(DATA_DIR):
            shutil.rmtree(DATA_DIR)
            os.makedirs(DATA_DIR, exist_ok=True)

        overall_end_time = time.time()
        print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))
        print("TOTAL TIME list pdfs:",  pipeline_times['list'])
        print("TOTAL TIME download pdfs:",  pipeline_times['download'])
        print("TOTAL TIME pdf -> txt and img time:",  pipeline_times['pdf_to_txt_img'])
        print("TOTAL TIME txt -> embed time:", pipeline_times['text_embed_time'])
        print("TOTAL TIME img -> embed time:", pipeline_times['img_embed_time'])
        print("TOTAL TIME metadata time:", pipeline_times['metadata_time'])
        print("TOTAL TIME uploading data:", pipeline_times['upload'])

    def main():
        batched_file_download(BATCH_SIZE, processor) 

    main()
    text_model.model.stop_multi_process_pool(model_pool)
