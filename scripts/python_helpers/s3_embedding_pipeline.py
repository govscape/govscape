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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count, get_context

# ****************************************************************************************************
# to run this file: poetry run python s3_ec2_embedding_pipeline.py 
# ****************************************************************************************************

def download_pdfs(pdfs, bucket_name, pdf_directory):
    s3 = boto3.client("s3")
    downloaded_files = []
    pdfs_downloaded = 0
    for pdf in pdfs:
        file_name = os.path.basename(pdf)
        local_path = os.path.join(pdf_directory, file_name)
        s3.download_file(bucket_name, pdf, local_path)
        downloaded_files.append(local_path)
        pdfs_downloaded += 1
    return downloaded_files

# ****************************************************************************************************
# gets pdfs from s3
def list_pdfs(num_pages, progress_path, bucket_name, pdfs_dir, num_servers, server_id):
    s3 = boto3.client("s3")
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
        pdf_keys = [key for key in pdf_keys if (hash(key) % num_servers) == server_id]

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
def upload_directory_to_s3(ec2_dir, bucket_name, s3_dir):
    subprocess.run(f"/home/ubuntu/.local/bin/s5cmd --log error cp {ec2_dir} s3://{bucket_name}/{s3_dir}".split())

# processing the pdfs: running through embedding pipeline and uploading to s3
def process_pdfs(pdf_files, processor, do_text_embedding, do_img_embedding, do_metadata_collection, pipeline_times,
                    data_dir_s3, bucket_name, local_data_dir):
    print("Do_Text_embedding: ", do_text_embedding)
    print("Do_Img_embedding: ", do_img_embedding)
    print("Do_Metadata_collection: ", do_metadata_collection)

    txt_directory = os.path.join(local_data_dir, 'txt')
    image_directory = os.path.join(local_data_dir, 'img')
    embeddings_directory = os.path.join(local_data_dir, 'embeddings')
    embeddings_img_pg_directory = os.path.join(local_data_dir, 'embeddings_img_pg')
    metadata_dir = os.path.join(local_data_dir, 'metadata')

    # PROCESS PDFS HERE 
    pdf_to_txt_img, text_embed_time, img_embed_time, metadata_time = processor.pdfs_to_embeddings(pdf_files,
                                                                                                    do_text_embedding,
                                                                                                    do_img_embedding,
                                                                                                    do_metadata_collection)
    pipeline_times['pdf_to_txt_img'] += pdf_to_txt_img
    pipeline_times['text_embed_time'] += text_embed_time
    pipeline_times['img_embed_time'] += img_embed_time 
    pipeline_times['metadata_time'] += metadata_time

    time1 = time.time()
    # UPLOADING EMBEDDINGS, TXTS, IMAGES TO S3 HERE 
    if do_text_embedding or do_img_embedding:
        upload_directory_to_s3(txt_directory, bucket_name, data_dir_s3)
        print("finished uploading txt")
        upload_directory_to_s3(image_directory, bucket_name, data_dir_s3)
        print("finished uploading img")
    if do_text_embedding:
        upload_directory_to_s3(embeddings_directory, bucket_name, data_dir_s3)
        print("finished uploading embeddings")
    if do_img_embedding:
        upload_directory_to_s3(embeddings_img_pg_directory, bucket_name, data_dir_s3)
        print("finished uploading embed img pg")
    if do_metadata_collection:
        upload_directory_to_s3(metadata_dir, bucket_name, data_dir_s3)
        print("finished uploading metadata")

    time2 = time.time()

    pipeline_times['upload'] += time2-time1
    pipeline_times['pdfs_processed'] += len(pdf_files)

    print("finished uploading current batch")
    print("pipeline times: ", pipeline_times)

# Fix for annoying argparse behavior with booleans
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # overall method that gets the files in batches and runs them through the pipeline
    def main():
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
        parser.add_argument('--do_text_embedding', type=str2bool, help='Whether to do text embedding', default=True)
        parser.add_argument('--do_img_embedding', type=str2bool, help='Whether to do image embedding', default=True)
        parser.add_argument('--do_metadata_collection', type=str2bool, help='Whether to do metadata collection', default=True)
        args = parser.parse_args()
        print("Arguments: ", args)
        NUM_PAGES_TO_PROCESS = args.num_pages_to_process
        BATCH_SIZE = args.batch_size

        # s3://bcgl-public-bucket/2008_EOT_PDFs/PDFs/
        bucket_name = args.bucket_name # 'bcgl-public-bucket'
        pdfs_dir = args.pdf_dir # 'archive/2020/PDFs/'# INPUT DATA DIR IN S3 HERE 
        data_dir_s3 = args.data_dir # 'prod-serving/' # OUTPUT OVERALL DATA DIR IN S3 HERE 

        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        local_data_dir = os.path.join(PROJECT_ROOT, 'data', 'prod')
        pdf_directory = os.path.join(local_data_dir, 'PDFs')
        progress_path = 'progress.json'  # Token to track of which pages have already been processed

        # ****************************************************************************************************
        pipeline_times = {'list' : 0, 'download' : 0, 'pdf_to_txt_img': 0, 'text_embed_time': 0, 'img_embed_time': 0, 'metadata_time': 0, 'upload' : 0, 'pdfs_processed' : 0}  # to keep track of the time it takes for each step in the pipeline

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
        processor = gs.PDFsToEmbeddings(pdf_directory, local_data_dir, text_model, model_pool)
        
        overall_start_time = time.time()

        # get the pdf files from s3
        time_list = time.time()
        pdf_files = list_pdfs(NUM_PAGES_TO_PROCESS, progress_path, bucket_name, pdfs_dir, args.num_servers, args.server_id)
        pipeline_times['list'] = time.time() - time_list

        print("Now starting with total number of PDF files: ", len(pdf_files))

        for i in range(0, len(pdf_files), BATCH_SIZE):
            print('*****************************************************************************************************')
            print("WE ARE ON BATCH: ", i)
            print('*****************************************************************************************************')
            batch = pdf_files[i:i + BATCH_SIZE] 
            local_batch = []
            time_download = time.time()
            os.makedirs(pdf_directory, exist_ok=True)
            download_batch_size = 100
            download_batches = [batch[i:i + download_batch_size] for i in range(0, len(batch), download_batch_size)]
            with get_context("fork").Pool(processes=os.cpu_count()*2) as pool:
                results = pool.starmap(
                    download_pdfs,
                    [(pdfs, bucket_name, pdf_directory) for pdfs in download_batches]
                )
                for file_names in results:
                    local_batch.extend(file_names)
            pipeline_times['download'] += time.time() - time_download
            print("len(local_batch) = ", len(local_batch))

            process_pdfs(local_batch, processor, args.do_text_embedding, args.do_img_embedding, args.do_metadata_collection, pipeline_times,
                            data_dir_s3, bucket_name, local_data_dir)

            if os.path.exists(local_data_dir):
                if args.do_text_embedding or args.do_img_embedding:
                    shutil.rmtree(local_data_dir + "/txt")
                    shutil.rmtree(local_data_dir + "/img")
                if args.do_text_embedding:
                    shutil.rmtree(local_data_dir + "/embeddings")
                if args.do_img_embedding:
                    shutil.rmtree(local_data_dir + "/embeddings_img_pg")
                if args.do_metadata_collection:
                    shutil.rmtree(local_data_dir + "/metadata")
                os.makedirs(local_data_dir, exist_ok=True)

            if os.path.exists(pdf_directory):
                shutil.rmtree(pdf_directory)
                os.makedirs(pdf_directory, exist_ok=True)

            # Write pipeline_times to a JSON file
            perf_filename = f"performance_{args.server_id}.json"
            perf_path = os.path.join(local_data_dir, perf_filename)
            with open(perf_path, "w") as f:
                json.dump(pipeline_times, f, indent=2)

            # Upload the performance JSON to S3
            s3 = boto3.client("s3")
            s3.upload_file(perf_path, bucket_name, os.path.join(data_dir_s3, perf_filename))
        
        
        # After all batches are processed, clean up the directories
        if os.path.exists(local_data_dir):
            shutil.rmtree(local_data_dir)
            os.makedirs(local_data_dir, exist_ok=True)

        overall_end_time = time.time()
        print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))
        print("TOTAL TIME list pdfs:",  pipeline_times['list'])
        print("TOTAL TIME download pdfs:",  pipeline_times['download'])
        print("TOTAL TIME pdf -> txt and img time:",  pipeline_times['pdf_to_txt_img'])
        print("TOTAL TIME txt -> embed time:", pipeline_times['text_embed_time'])
        print("TOTAL TIME img -> embed time:", pipeline_times['img_embed_time'])
        print("TOTAL TIME metadata time:", pipeline_times['metadata_time'])
        print("TOTAL TIME uploading data:", pipeline_times['upload'])
        text_model.model.stop_multi_process_pool(model_pool)

    main()
