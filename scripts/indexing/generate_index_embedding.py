import json
import logging
import os
import shutil
import time

import numpy as np

from govscape.config import DataModel
from govscape.data_loader import RemoteDirectoryIterator, build_data_loader
from govscape.utils import base_argument_parser

import govscape as gs

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

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
    parser = base_argument_parser(description="Generate embedding index")
    parser.set_defaults(batch_size=350000)
    parser.add_argument(
        "--embedding_type",
        type=str,
        help="Which embedding type to index [txt, img_pg]",
        required=True,
    )
    parser.add_argument(
        "--index_type",
        type=str,
        choices=["FAISS"],
        help='Type of index to create (e.g., "FAISS")',
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
    local_dm = DataModel(LOCAL_DATA_DIR)
    remote_dm = DataModel(REMOTE_DATA_DIR)
    REMOTE_EMBEDDING_DIR, REMOTE_INDEX_DIR, LOCAL_EMBEDDING_DIR, LOCAL_INDEX_DIR = (
        None,
        None,
        None,
        None,
    )
    print("Embedding type specified: ", args.embedding_type)
    print(args.embedding_type == "txt")
    if args.embedding_type == "txt":
        REMOTE_EMBEDDING_DIR = (
            remote_dm.embedding_directory
        )  # 'prod-serving/embeddings/'
        REMOTE_INDEX_DIR = remote_dm.index_directory  # 'prod-serving/index'
        LOCAL_EMBEDDING_DIR = (
            local_dm.embedding_directory
        )  # 'govscape/data/prod/embeddings/'
        LOCAL_INDEX_DIR = local_dm.index_directory  # 'govscape/data/prod/index/'
    elif args.embedding_type == "img_pg":
        REMOTE_EMBEDDING_DIR = (
            remote_dm.embedding_img_pg_directory
        )  # 'prod-serving/embeddings_img_pg/'
        REMOTE_INDEX_DIR = (
            remote_dm.index_img_pg_directory
        )  # 'prod-serving/index_img_pg'
        LOCAL_EMBEDDING_DIR = (
            local_dm.embedding_img_pg_directory
        )  # 'govscape/data/prod/embeddings_img_pg/'
        LOCAL_INDEX_DIR = (
            local_dm.index_img_pg_directory
        )  # 'govscape/data/prod/index_img_pg/'
    REMOTE_CHECKPOINT_PATH = os.path.join(
        remote_dm.checkpoints_directory,
        "checkpoint_index_" + args.embedding_type + ".json",
    )
    LOCAL_CHECKPOINT_PATH = os.path.join(
        local_dm.checkpoints_directory,
        "checkpoint_index_" + args.embedding_type + ".json",
    )
    REMOTE_PERFORMANCE_PATH = os.path.join(
        remote_dm.performance_directory,
        "performance_index_" + args.embedding_type + ".json",
    )
    LOCAL_PERFORMANCE_PATH = os.path.join(
        local_dm.performance_directory,
        "performance_index_" + args.embedding_type + ".json",
    )

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(LOCAL_EMBEDDING_DIR, exist_ok=True)
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)
    os.makedirs(local_dm.checkpoints_directory, exist_ok=True)
    os.makedirs(local_dm.performance_directory, exist_ok=True)

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
        index.add_batch(embeddings, names, pages)
        index.save_index()

        pipeline_times["embedding_indexing_time"] += time.time() - time_index_start

        time1 = time.time()

        # UPLOADING Indexes TO S3 HERE
        data_loader.upload_directory(LOCAL_INDEX_DIR, REMOTE_INDEX_DIR)
        print("finished uploading index")
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
