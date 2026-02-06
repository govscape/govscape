import os
import argparse
import time
import govscape as gs
import torch
import shutil
import json

# ****************************************************************************************************
# to run this file: poetry run python s3_ec2_embedding_pipeline.py 
# ****************************************************************************************************

# uploads dir of files to s3
def upload_directory_to_backend(data_loader, local_dir, remote_dir):
    data_loader.upload_directory(local_dir, remote_dir)

# processing the pdfs: running through embedding pipeline and uploading to s3
def process_pdfs(pdf_files, processor, do_text_embedding, do_img_embedding, do_metadata_collection, pipeline_times,
                    data_dir_backend, data_loader, local_data_dir):
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
        upload_directory_to_backend(data_loader, txt_directory, data_dir_backend)
        print("finished uploading txt")
        upload_directory_to_backend(data_loader, image_directory, data_dir_backend)
        print("finished uploading img")
    if do_text_embedding:
        upload_directory_to_backend(data_loader, embeddings_directory, data_dir_backend)
        print("finished uploading embeddings")
    if do_img_embedding:
        upload_directory_to_backend(data_loader, embeddings_img_pg_directory, data_dir_backend)
        print("finished uploading embed img pg")
    if do_metadata_collection:
        upload_directory_to_backend(data_loader, metadata_dir, data_dir_backend)
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
        parser.add_argument('--text_model_type', type=str, help='The model type to use for text embedding', default='ST')
        parser.add_argument('--visual_model_type', type=str, help='The model type to use for visual embedding', default='CLIP')
        parser.add_argument('--num_servers', type=int, help='The number of servers to use for embedding', default=1)
        parser.add_argument('--server_id', type=int, help='The ID of the current server', default=0)
        parser.add_argument('--do_text_embedding', type=str2bool, help='Whether to do text embedding', default=True)
        parser.add_argument('--do_img_embedding', type=str2bool, help='Whether to do image embedding', default=True)
        parser.add_argument('--do_metadata_collection', type=str2bool, help='Whether to do metadata collection', default=True)
        parser.add_argument('--backend', choices=['s3', 'local'], default='s3', help='Data backend to use')
        parser.add_argument('--bucket_name', type=str, help='S3 Bucket Name')
        parser.add_argument('--local_base_dir', type=str, default='data', help='Base directory for local backend')
        parser.add_argument('--pdf_dir', type=str, help='Directory containing PDFs')
        parser.add_argument('--data_dir', type=str, help='Directory for output data')
        args = parser.parse_args()
        NUM_PAGES_TO_PROCESS = args.num_pages_to_process
        BATCH_SIZE = args.batch_size

        bucket_name = args.bucket_name # 'bcgl-public-bucket'
        pdfs_dir = args.pdf_dir # 'archive/2020/PDFs/'# INPUT DATA DIR HERE 
        data_dir_backend = args.data_dir # 'prod-serving/' # OUTPUT OVERALL DATA DIR HERE 

        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        local_data_dir = os.path.join(PROJECT_ROOT, 'data', 'prod')
        pdf_directory = os.path.join(local_data_dir, 'PDFs')
        progress_path = 'progress.json'  # Token to track of which pages have already been processed

        # ****************************************************************************************************
        pipeline_times = {'list' : 0, 'download' : 0, 'pdf_to_txt_img': 0, 'text_embed_time': 0, 'img_embed_time': 0, 'metadata_time': 0, 'upload' : 0, 'pdfs_processed' : 0}  # to keep track of the time it takes for each step in the pipeline

        data_loader = gs.build_data_loader(
            args.backend,
            bucket_name,
            args.local_base_dir,
            checkpoint_path=progress_path,
        )

        processor = gs.PDFsToEmbeddings(pdf_directory, local_data_dir, args.text_model_type, args.visual_model_type)
        
        overall_start_time = time.time()

        max_files_to_process = NUM_PAGES_TO_PROCESS * 1000
        files_processed = 0
        while files_processed < max_files_to_process:
            print('*****************************************************************************************************')
            print("FILES PROCESSED: ", files_processed)
            print('*****************************************************************************************************')
            time_download = time.time()
            os.makedirs(pdf_directory, exist_ok=True)
            batch_limit = min(BATCH_SIZE, max_files_to_process - files_processed)
            local_batch = data_loader.download_files(
                pdfs_dir,
                pdf_directory,
                max_keys=batch_limit,
                filter_fn=lambda key: key.endswith('.pdf')
                and (hash(key) % args.num_servers) == args.server_id,
            )
            pipeline_times['download'] += time.time() - time_download
            if not local_batch:
                break
            print("len(local_batch) = ", len(local_batch))

            process_pdfs(
                local_batch,
                processor,
                args.do_text_embedding,
                args.do_img_embedding,
                args.do_metadata_collection,
                pipeline_times,
                data_dir_backend,
                data_loader,
                local_data_dir,
            )

            data_loader.save_checkpoint()
            data_loader.save_checkpoint()
            files_processed += len(local_batch)

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
            data_loader.upload_file(perf_path, os.path.join(data_dir_backend, perf_filename))
        
        
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

    main()
