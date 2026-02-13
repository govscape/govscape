import argparse
import json
import os
import shutil
import time

from govscape.data_loader import RemoteDirectoryIterator, build_data_loader

import govscape as gs

# ---------------------------------------------------------------------------
# to run this file: poetry run python s3_ec2_embedding_pipeline.py
# ---------------------------------------------------------------------------


# uploads dir of files to s3
def upload_directory_to_backend(data_loader, local_dir, remote_dir):
    data_loader.upload_directory(local_dir, remote_dir)


# processing the pdfs: running through embedding pipeline and uploading to s3
def process_pdfs(
    pdf_files,
    processor,
    do_text_embedding,
    do_img_embedding,
    do_metadata_collection,
    pipeline_times,
    data_dir_backend,
    data_loader,
    local_data_dir,
):
    print("Do_Text_embedding: ", do_text_embedding)
    print("Do_Img_embedding: ", do_img_embedding)
    print("Do_Metadata_collection: ", do_metadata_collection)

    txt_directory = os.path.join(local_data_dir, "txt")
    image_directory = os.path.join(local_data_dir, "img")
    embeddings_directory = os.path.join(local_data_dir, "embeddings")
    embeddings_img_pg_directory = os.path.join(local_data_dir, "embeddings_img_pg")
    metadata_dir = os.path.join(local_data_dir, "metadata")

    # PROCESS PDFS HERE
    pdf_to_txt_img_time, text_embed_time, img_embed_time, metadata_time = (
        processor.pdfs_to_embeddings(
            pdf_files, do_text_embedding, do_img_embedding, do_metadata_collection
        )
    )
    pipeline_times["pdf_to_txt_img_time"] += pdf_to_txt_img_time
    pipeline_times["text_embed_time"] += text_embed_time
    pipeline_times["img_embed_time"] += img_embed_time
    pipeline_times["metadata_time"] += metadata_time

    time1 = time.time()
    # UPLOADING EMBEDDINGS, TXTS, IMAGES TO S3 HERE
    if do_text_embedding or do_img_embedding:
        upload_directory_to_backend(
            data_loader, txt_directory, os.path.join(data_dir_backend, "txt")
        )
        print("finished uploading txt")
        upload_directory_to_backend(
            data_loader, image_directory, os.path.join(data_dir_backend, "img")
        )
        print("finished uploading img")
    if do_text_embedding:
        upload_directory_to_backend(
            data_loader,
            embeddings_directory,
            os.path.join(data_dir_backend, "embeddings"),
        )
        print("finished uploading embeddings")
    if do_img_embedding:
        upload_directory_to_backend(
            data_loader,
            embeddings_img_pg_directory,
            os.path.join(data_dir_backend, "embeddings_img_pg"),
        )
        print("finished uploading embed img pg")
    if do_metadata_collection:
        upload_directory_to_backend(
            data_loader, metadata_dir, os.path.join(data_dir_backend, "metadata")
        )
        print("finished uploading metadata")

    time2 = time.time()

    pipeline_times["upload"] += time2 - time1
    pipeline_times["pdfs_processed"] += len(pdf_files)

    print("finished uploading current batch")
    print("pipeline times: ", pipeline_times)


# Fix for annoying argparse behavior with booleans
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    # overall method that gets the files in batches and runs them through the
    # pipeline
    def main():
        # FIELDS TO SET --------------------------------------------------------
        parser = argparse.ArgumentParser(description="S3 EC2 Embedding Pipeline")
        parser.add_argument(
            "--num_pages_to_process",
            type=int,
            default=100,
            help="Number of pages to process from S3",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=1000,
            help="Number of pdfs to process at a time",
        )
        parser.add_argument(
            "--text_model_type",
            type=str,
            help="The model type to use for text embedding",
            default="ST",
        )
        parser.add_argument(
            "--visual_model_type",
            type=str,
            help="The model type to use for visual embedding",
            default="CLIP",
        )
        parser.add_argument(
            "--num_servers",
            type=int,
            help="The number of servers to use for embedding",
            default=1,
        )
        parser.add_argument(
            "--server_id", type=int, help="The ID of the current server", default=0
        )
        parser.add_argument(
            "--do_text_embedding",
            type=str2bool,
            help="Whether to do text embedding",
            default=True,
        )
        parser.add_argument(
            "--do_img_embedding",
            type=str2bool,
            help="Whether to do image embedding",
            default=True,
        )
        parser.add_argument(
            "--do_metadata_collection",
            type=str2bool,
            help="Whether to do metadata collection",
            default=True,
        )
        parser.add_argument(
            "--backend",
            choices=["s3", "local"],
            default="s3",
            help="Data backend to use",
        )
        parser.add_argument("--bucket_name", type=str, help="S3 Bucket Name")
        parser.add_argument(
            "--local_base_dir",
            type=str,
            default="data",
            help="Base directory for local backend",
        )
        parser.add_argument("--pdf_dir", type=str, help="Directory containing PDFs")
        parser.add_argument(
            "--remote_data_dir", type=str, help="Directory for output data"
        )
        args = parser.parse_args()
        NUM_PAGES_TO_PROCESS = args.num_pages_to_process
        BATCH_SIZE = args.batch_size

        # ---------------------------------------------------------------------------
        BUCKET_NAME = args.bucket_name
        REMOTE_PDF_DIR = args.pdf_dir  # e.g. "archive/2020/PDFs/"
        REMOTE_DATA_DIR = args.remote_data_dir  # e.g. "prod-serving"
        PROJECT_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )  # e.g. "govscape"
        LOCAL_DATA_DIR = os.path.join(
            PROJECT_ROOT, "data", "prod"
        )  # "govscape/data/prod"
        LOCAL_PDF_DIR = os.path.join(
            LOCAL_DATA_DIR, "PDFs"
        )  # e.g. "govscape/data/prod/PDFs"
        LOCAL_TXT_DIR = os.path.join(
            LOCAL_DATA_DIR, "txt"
        )  # e.g. "govscape/data/prod/txt"
        LOCAL_IMG_DIR = os.path.join(
            LOCAL_DATA_DIR, "img"
        )  # e.g. "govscape/data/prod/img"
        LOCAL_EMBEDDINGS_DIR = os.path.join(
            LOCAL_DATA_DIR, "embeddings"
        )  # e.g. "govscape/data/prod/embeddings"
        LOCAL_EMBEDDINGS_IMG_PG_DIR = os.path.join(
            LOCAL_DATA_DIR, "embeddings_img_pg"
        )  # e.g. "govscape/data/prod/embeddings_img_pg"
        LOCAL_METADATA_DIR = os.path.join(
            LOCAL_DATA_DIR, "metadata"
        )  # e.g. "govscape/data/prod/metadata"
        LOCAL_CHECKPOINT_PATH = os.path.join(
            LOCAL_DATA_DIR,
            "checkpoints",
            f"checkpoint_embedding_pipeline_{args.server_id}.json",
        )
        # e.g. "govscape/data/prod/checkpoints/checkpoint_embedding_pipeline_0.json"
        REMOTE_CHECKPOINT_PATH = os.path.join(
            REMOTE_DATA_DIR,
            "checkpoints",
            f"checkpoint_embedding_pipeline_{args.server_id}.json",
        )
        # e.g. "prod-serving/checkpoints/checkpoint_embedding_pipeline_0.json"
        LOCAL_PERF_PATH = os.path.join(
            LOCAL_DATA_DIR, "performance", f"performance_{args.server_id}.json"
        )
        # e.g. "govscape/data/prod/performance/performance_0.json"
        REMOTE_PERF_PATH = os.path.join(
            REMOTE_DATA_DIR, "performance", f"performance_{args.server_id}.json"
        )
        # e.g. "prod-serving/performance/performance_0.json"

        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
        os.makedirs(LOCAL_PDF_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(LOCAL_CHECKPOINT_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(LOCAL_PERF_PATH), exist_ok=True)

        # ---------------------------------------------------------------------------
        pipeline_times = {
            "list": 0,
            "download": 0,
            "pdf_to_txt_img_time": 0,
            "text_embed_time": 0,
            "img_embed_time": 0,
            "metadata_time": 0,
            "upload": 0,
            "pdfs_processed": 0,
        }
        # to keep track of the time it takes for each step in the pipeline

        data_loader = build_data_loader(
            args.backend,
            BUCKET_NAME,
            local_base_dir=args.local_base_dir,
        )
        remote_pdf_iter = RemoteDirectoryIterator(
            data_loader,
            REMOTE_PDF_DIR,
            remote_checkpoint_path=REMOTE_CHECKPOINT_PATH,
            local_checkpoint_path=LOCAL_CHECKPOINT_PATH,
            local_dir=LOCAL_PDF_DIR,
        )

        processor = gs.PDFsToEmbeddings(
            LOCAL_PDF_DIR, LOCAL_DATA_DIR, args.text_model_type, args.visual_model_type
        )

        overall_start_time = time.time()

        max_files_to_process = NUM_PAGES_TO_PROCESS * 1000
        files_processed = 0
        while files_processed < max_files_to_process:
            print("-" * 93)
            print("FILES PROCESSED: ", files_processed)
            print("-" * 93)
            time_download = time.time()
            os.makedirs(LOCAL_PDF_DIR, exist_ok=True)
            batch_limit = min(BATCH_SIZE, max_files_to_process - files_processed)
            local_batch = remote_pdf_iter.download_batch(
                max_keys=batch_limit,
                filter_fn=lambda key: key.endswith(".pdf")
                and (hash(key) % args.num_servers) == args.server_id,
            )
            pipeline_times["download"] += time.time() - time_download
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
                REMOTE_DATA_DIR,
                data_loader,
                LOCAL_DATA_DIR,
            )

            remote_pdf_iter.save_checkpoint()
            files_processed += len(local_batch)

            if os.path.exists(LOCAL_DATA_DIR):
                if args.do_text_embedding or args.do_img_embedding:
                    if os.path.exists(LOCAL_TXT_DIR):
                        shutil.rmtree(LOCAL_TXT_DIR)
                    if os.path.exists(LOCAL_IMG_DIR):
                        shutil.rmtree(LOCAL_IMG_DIR)
                if args.do_text_embedding and os.path.exists(LOCAL_EMBEDDINGS_DIR):
                    shutil.rmtree(LOCAL_EMBEDDINGS_DIR)
                if args.do_img_embedding and os.path.exists(
                    LOCAL_EMBEDDINGS_IMG_PG_DIR
                ):
                    shutil.rmtree(LOCAL_EMBEDDINGS_IMG_PG_DIR)
                if args.do_metadata_collection and os.path.exists(LOCAL_METADATA_DIR):
                    shutil.rmtree(LOCAL_METADATA_DIR)
                os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

            if os.path.exists(LOCAL_PDF_DIR):
                shutil.rmtree(LOCAL_PDF_DIR)
                os.makedirs(LOCAL_PDF_DIR, exist_ok=True)

            # Write pipeline_times to a JSON file
            with open(LOCAL_PERF_PATH, "w") as f:
                json.dump(pipeline_times, f, indent=2)

            # Upload the performance JSON to S3
            data_loader.upload_file(LOCAL_PERF_PATH, REMOTE_PERF_PATH)

        # After all batches are processed, clean up the directories
        if os.path.exists(LOCAL_DATA_DIR):
            shutil.rmtree(LOCAL_DATA_DIR)
            os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

        overall_end_time = time.time()
        print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))
        print("TOTAL TIME list pdfs:", pipeline_times["list"])
        print("TOTAL TIME download pdfs:", pipeline_times["download"])
        print(
            "TOTAL TIME pdf -> txt and img time:", pipeline_times["pdf_to_txt_img_time"]
        )
        print("TOTAL TIME txt -> embed time:", pipeline_times["text_embed_time"])
        print("TOTAL TIME img -> embed time:", pipeline_times["img_embed_time"])
        print("TOTAL TIME metadata time:", pipeline_times["metadata_time"])
        print("TOTAL TIME uploading data:", pipeline_times["upload"])

    main()
