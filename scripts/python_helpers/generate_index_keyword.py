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
        default=100000,
        help="Number of pages to process at a time",
    )
    parser.add_argument("--bucket_name", type=str, help="S3 Bucket Name")
    parser.add_argument("--remote_data_dir", type=str, help="Remote Data Directory")
    parser.add_argument(
        "--keyword_index_type",
        type=str,
        default="LanceDB",
        help="Type of keyword index to use: LanceDB, SQLite or Whoosh",
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
    INDEX_TYPE = args.keyword_index_type  # 'LanceDB', 'SQLite' or 'Whoosh'

    # ---------------------------------------------------------------------------
    BUCKET_NAME = args.bucket_name  # 'bcgl-public-bucket'
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "prod")
    REMOTE_DATA_DIR = args.remote_data_dir  # 'prod-serving/'
    REMOTE_TXT_DIR = os.path.join(REMOTE_DATA_DIR, "txt")
    LOCAL_TXT_DIR = os.path.join(LOCAL_DATA_DIR, "txt")
    REMOTE_INDEX_PREFIX = "index_keyword"
    REMOTE_INDEX_DIR = os.path.join(REMOTE_DATA_DIR, REMOTE_INDEX_PREFIX)
    LOCAL_INDEX_DIR = os.path.join(LOCAL_DATA_DIR, REMOTE_INDEX_PREFIX)
    REMOTE_CHECKPOINT_PATH = os.path.join(REMOTE_DATA_DIR, "checkpoints", "checkpoint_text_index.json")
    LOCAL_CHECKPOINT_PATH = os.path.join(
        LOCAL_DATA_DIR, "checkpoints", "checkpoint_text_index.json"
    )
    REMOTE_PERFORMANCE_PATH = os.path.join(
        REMOTE_DATA_DIR, "performance", "performance_keyword_index.json"
    )
    LOCAL_PERFORMANCE_PATH = os.path.join(
        LOCAL_DATA_DIR, "performance", "performance_keyword_index.json"
    )

    if os.path.isdir(LOCAL_DATA_DIR):
        shutil.rmtree(LOCAL_DATA_DIR, ignore_errors=True)

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(LOCAL_TXT_DIR, exist_ok=True)
    os.makedirs(LOCAL_INDEX_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOCAL_CHECKPOINT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOCAL_PERFORMANCE_PATH), exist_ok=True)

    # ---------------------------------------------------------------------------
    pipeline_times = {
        "list": 0,
        "download": 0,
        "keyword_indexing_time": 0,
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
        REMOTE_TXT_DIR,
        remote_checkpoint_path=REMOTE_CHECKPOINT_PATH,
        local_checkpoint_path=LOCAL_CHECKPOINT_PATH,
        local_dir=LOCAL_TXT_DIR,
    )

    remote_existing_idx_files = data_loader.list_objects(REMOTE_INDEX_DIR)
    for remote_file in remote_existing_idx_files.keys:
        data_loader.download_file(
            remote_file, os.path.join(LOCAL_INDEX_DIR, os.path.basename(remote_file))
        )

    # uploads dir of files to backend
    def upload_directory_to_backend(local_dir, remote_dir):
        data_loader.upload_directory(local_dir, remote_dir)

    # process txts: update index and upload to backend
    def process_txt_files(txt_files):
        time_index_start = time.time()
        if INDEX_TYPE == "LanceDB":
            index = gs.LanceDBKeywordIndex(LOCAL_INDEX_DIR)
        elif INDEX_TYPE == "SQLite":
            index = gs.SQLiteKeywordIndex(LOCAL_INDEX_DIR)
        elif INDEX_TYPE == "Whoosh":
            index = gs.WhooshKeywordIndex(LOCAL_INDEX_DIR)
        else:
            raise ValueError(
                "index_type must be either 'LanceDB', 'SQLite', or 'Whoosh'"
            )

        index.load_index()
        names = []
        pages = []
        txts = []
        for txt_file in txt_files:
            txt_file_path = os.path.join(LOCAL_TXT_DIR, txt_file)
            if not os.path.exists(txt_file_path):
                print(f"File {txt_file_path} does not exist. Skipping.")
                continue
            names.append(os.path.basename(os.path.dirname(txt_file_path)))
            pages.append(txt_file_path.replace(".txt", "").rpartition("_")[2])
            txt = None
            with open(txt_file_path) as f:
                txt = f.read()
            txts.append(txt)
        index.add_batch(txts, names, pages)
        index.save_index()

        pipeline_times["keyword_indexing_time"] += time.time() - time_index_start

        time1 = time.time()
        # UPLOADING Indexes TO S3 HERE
        data_loader.upload_directory(LOCAL_INDEX_DIR, REMOTE_INDEX_DIR)
        print("finished uploading keyword index")
        time2 = time.time()

        pipeline_times["upload"] += time2 - time1
        pipeline_times["pdfs_processed"] += len(txt_files)

    # overall method that gets the files in batches and runs them through the
    # pipeline
    def batched_file_download(BATCH_SIZE):
        try:
            data_loader.download_file(REMOTE_PERFORMANCE_PATH, LOCAL_PERFORMANCE_PATH)
            with open(LOCAL_PERFORMANCE_PATH) as f:
                existing_pipeline_times = json.load(f)
                for key in pipeline_times:
                    pipeline_times[key] = existing_pipeline_times.get(key, 0)
        except Exception:
            print("No existing performance file found. Starting fresh.")
        overall_start_time = time.time()
        files_processed = 0
        max_files_to_process = NUM_PAGES_TO_PROCESS * 1000
        while files_processed < max_files_to_process:
            print("-" * 93)
            print("FILES PROCESSED: ", files_processed)
            print("-" * 93)

            time_download = time.time()
            batch_limit = min(BATCH_SIZE, max_files_to_process - files_processed)
            local_paths = remote_iter.download_batch(
                max_keys=batch_limit,
                filter_fn=lambda key: key.endswith(".txt"),
            )
            pipeline_times["download"] += time.time() - time_download

            if not local_paths:
                break

            local_batch = [
                os.path.relpath(path, LOCAL_TXT_DIR).replace("\\", "/")
                for path in local_paths
            ]
            process_txt_files(local_batch)

            # Write continuation token to progress file
            remote_iter.save_checkpoint()

            # Write pipeline_times to a JSON file
            with open(LOCAL_PERFORMANCE_PATH, "w") as f:
                json.dump(pipeline_times, f, indent=2)

            # Upload the performance json to backend
            data_loader.upload_file(LOCAL_PERFORMANCE_PATH, REMOTE_PERFORMANCE_PATH)
            print("finished uploading current batch")
            print("pipeline times: ", pipeline_times)
            files_processed += len(local_paths)

            # delete the directories except for the indices which will continue to be
            # updated
            if os.path.exists(LOCAL_TXT_DIR):
                shutil.rmtree(LOCAL_TXT_DIR)
                os.makedirs(LOCAL_TXT_DIR, exist_ok=True)

        # After all batches are processed, clean up the directories
        if os.path.exists(LOCAL_TXT_DIR):
            shutil.rmtree(LOCAL_TXT_DIR)
        if os.path.exists(LOCAL_INDEX_DIR):
            shutil.rmtree(LOCAL_INDEX_DIR)

        overall_end_time = time.time()
        print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))

    def main():
        batched_file_download(BATCH_SIZE)

    main()
