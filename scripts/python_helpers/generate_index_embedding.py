# AI modified: 2026-04-06 00:10:53 434ce298
import argparse
import json
import os
import shutil
import time

import numpy as np

from govscape.data_loader import RemoteDirectoryIterator, build_data_loader

import govscape as gs

# ---------------------------------------------------------------------------
# to run this file: poetry run python generate_index_embedding.py
# This file takes the output from the embedding pipeline (npy files) and
# creates an index using those embeddings. It then uploads the index to S3.
# The script is designed to run on EC2 and can process files in batches. It
# keeps track of which files have been processed using a checkpointing system
# so it can resume where it left off if interrupted.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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
        default=350000,
        help="Number of pages to process at a time",
    )
    parser.add_argument("--bucket_name", type=str, help="S3 Bucket Name")
    parser.add_argument(
        "--remote_data_dir", type=str, help="Remote Directory for input data"
    )
    parser.add_argument(
        "--embedding_prefix", type=str, help="S3 Prefix for embedding files"
    )
    parser.add_argument("--out_index_prefix", type=str, help="S3 Prefix for index data")
    parser.add_argument(
        "--index_type",
        type=str,
        choices=["FAISS"],
        help='Type of index to create (e.g., "FAISS")',
    )
    parser.add_argument(
        "--backend", choices=["s3", "local"], default="s3", help="Data backend to use"
    )
    parser.add_argument(
        "--local_base_dir",
        type=str,
        default="data",
        help="Base directory for local backend",
    )
    args = parser.parse_args()
    NUM_PAGES_TO_PROCESS = args.num_pages_to_process
    BATCH_SIZE = args.batch_size
    INDEX_TYPE = args.index_type  # 'FAISS' # TYPE OF INDEX TO CREATE

    # ---------------------------------------------------------------------------
    # All Local and Remote Paths
    BUCKET_NAME = args.bucket_name  # 'bcgl-public-bucket'
    PROJECT_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )  # 'govscape/'
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "prod")  # 'govscape/data/prod/'
    REMOTE_DATA_DIR = args.remote_data_dir  # 'prod-serving/'
    REMOTE_EMBEDDING_DIR = os.path.join(
        REMOTE_DATA_DIR, args.embedding_prefix
    )  # 'prod-serving/embeddings/'
    LOCAL_EMBEDDING_DIR = os.path.join(
        LOCAL_DATA_DIR, args.embedding_prefix.replace("/", "")
    )  # 'govscape/data/prod/embeddings/'
    REMOTE_INDEX_PREFIX = args.out_index_prefix.rstrip("/")  # 'index', 'index_img_pg'
    REMOTE_INDEX_DIR = os.path.join(
        REMOTE_DATA_DIR, REMOTE_INDEX_PREFIX
    )  # 'prod-serving/index'
    LOCAL_INDEX_DIR = os.path.join(LOCAL_DATA_DIR, REMOTE_INDEX_PREFIX)
    # 'govscape/data/prod/index/'
    REMOTE_CHECKPOINT_PATH = os.path.join(
        REMOTE_DATA_DIR, "checkpoints", "checkpoint_" + REMOTE_INDEX_PREFIX + ".json"
    )  # 'prod-serving/checkpoints/index_checkpoint.json'
    LOCAL_CHECKPOINT_PATH = os.path.join(
        LOCAL_DATA_DIR, "checkpoints", "checkpoint_" + REMOTE_INDEX_PREFIX + ".json"
    )
    # 'govscape/data/prod/checkpoints/checkpoint_index.json'
    REMOTE_PERFORMANCE_PATH = os.path.join(
        REMOTE_DATA_DIR, "performance", "performance_" + REMOTE_INDEX_PREFIX + ".json"
    )
    # 'prod-serving/performance/index_performance.json'
    LOCAL_PERFORMANCE_PATH = os.path.join(
        LOCAL_DATA_DIR, "performance", "performance_" + REMOTE_INDEX_PREFIX + ".json"
    )
    # 'govscape/data/prod/performance/performance_index.json'
    REMOTE_METADATA_INDEX_DIR = os.path.join(REMOTE_DATA_DIR, "index_metadata")
    LOCAL_METADATA_INDEX_DIR = os.path.join(LOCAL_DATA_DIR, "index_metadata")
    REMOTE_METADATA_DB_PATH = os.path.join(REMOTE_METADATA_INDEX_DIR, "metadata.db")
    LOCAL_METADATA_DB_PATH = os.path.join(LOCAL_METADATA_INDEX_DIR, "metadata.db")

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(LOCAL_EMBEDDING_DIR, exist_ok=True)
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)
    os.makedirs(LOCAL_METADATA_INDEX_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOCAL_CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOCAL_PERFORMANCE_PATH), exist_ok=True)

    # ---------------------------------------------------------------------------
    pipeline_times = {
        "list": 0,
        "download": 0,
        "embedding_indexing_time": 0,
        "upload": 0,
        "pdfs_processed": 0,
    }

    data_loader = build_data_loader(
        args.backend,
        BUCKET_NAME,
        local_base_dir=args.local_base_dir,
    )
    remote_iter = RemoteDirectoryIterator(
        data_loader,
        REMOTE_EMBEDDING_DIR,
        remote_checkpoint_path=REMOTE_CHECKPOINT_PATH,
        local_checkpoint_path=LOCAL_CHECKPOINT_PATH,
        local_dir=LOCAL_EMBEDDING_DIR,
    )

    # Download existing index files from S3 to local directory to update them
    # with new data and re-upload.
    remote_existing_idx_files = data_loader.list_objects(REMOTE_INDEX_DIR)
    for remote_file in remote_existing_idx_files.keys:
        data_loader.download_file(
            remote_file, os.path.join(LOCAL_INDEX_DIR, os.path.basename(remote_file))
        )

    try:
        data_loader.download_file(REMOTE_METADATA_DB_PATH, LOCAL_METADATA_DB_PATH)
    except Exception as e:
        print(f"No existing metadata DB found for vector store: {e}")

    metadata_index = gs.SQLiteMetadataIndex(LOCAL_METADATA_INDEX_DIR)
    metadata_index.build_index()

    embedding_type = "visual" if "img_pg" in args.embedding_prefix else "textual"

    # Adding Embedding Files to the Index and Uploading to S3
    def process_embedding_files(embedding_files):
        time_index_start = time.time()
        index = gs.FAISSIndex(LOCAL_INDEX_DIR)
        index.load_index()
        names = []
        pages = []
        embeddings = []
        for embedding_file in embedding_files:
            embedding_file_path = os.path.join(LOCAL_EMBEDDING_DIR, embedding_file)
            if not os.path.exists(embedding_file_path):
                print(f"File {embedding_file_path} does not exist. Skipping.")
                continue
            names.append(os.path.basename(os.path.dirname(embedding_file_path)))
            pages.append(embedding_file_path.replace(".npy", "").rpartition("_")[2])
            embeddings.append(np.load(embedding_file_path))
        print(f"Adding {len(embeddings)} embeddings to the index.")
        embeddings = np.asarray(embeddings)
        metadata_index.upsert_vectors_batch(embedding_type, names, pages, embeddings)
        index.add_batch(embeddings, names, pages)
        index.save_index()
        metadata_index.save_index()

        pipeline_times["embedding_indexing_time"] += time.time() - time_index_start

        time1 = time.time()

        # UPLOADING Indexes TO S3 HERE
        data_loader.upload_directory(LOCAL_INDEX_DIR, REMOTE_INDEX_DIR)
        print("finished uploading index")
        data_loader.upload_file(LOCAL_METADATA_DB_PATH, REMOTE_METADATA_DB_PATH)
        print("finished uploading metadata vector store")
        time2 = time.time()

        pipeline_times["upload"] += time2 - time1
        pipeline_times["pdfs_processed"] += len(embedding_files)

        # Write pipeline_times to a JSON file
        with open(LOCAL_PERFORMANCE_PATH, "w") as f:
            json.dump(pipeline_times, f, indent=2)

        # Upload the performance JSON to S3
        data_loader.upload_file(LOCAL_PERFORMANCE_PATH, REMOTE_PERFORMANCE_PATH)
        print("finished uploading current batch")
        print("pipeline times: ", pipeline_times)

    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE):
        overall_start_time = time.time()
        # Progress checkpoint is now managed by DataLoader\

        # Each page of results corresponds to 1 gzipped batch file
        max_files_to_process = NUM_PAGES_TO_PROCESS
        files_processed = 0
        while files_processed < max_files_to_process:
            print("-" * 93)
            print("FILES PROCESSED: ", files_processed)
            print("-" * 93)

            time_download = time.time()
            batch_limit = min(BATCH_SIZE, max_files_to_process - files_processed)
            local_paths = remote_iter.download_batch(
                max_keys=batch_limit,
                filter_fn=lambda key: key.endswith(".npy"),
            )
            print(len(local_paths), " files downloaded in this batch.")
            pipeline_times["download"] += time.time() - time_download
            if len(local_paths) == 0:
                break

            successful_downloads = [
                os.path.relpath(path, LOCAL_EMBEDDING_DIR).replace("\\", "/")
                for path in local_paths
            ]
            process_embedding_files(successful_downloads)

            # delete the directories except for the indices which will continue to be
            # updated
            if os.path.exists(LOCAL_DATA_DIR):
                shutil.rmtree(LOCAL_EMBEDDING_DIR)
                os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

            files_processed += len(local_paths)
            remote_iter.save_checkpoint()

        # After all batches are processed, clean up the directories
        if os.path.exists(LOCAL_EMBEDDING_DIR):
            shutil.rmtree(LOCAL_EMBEDDING_DIR)
        if os.path.exists(LOCAL_INDEX_DIR):
            shutil.rmtree(LOCAL_INDEX_DIR)

        overall_end_time = time.time()
        print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))

    def main():
        batched_file_download(BATCH_SIZE)

    main()
