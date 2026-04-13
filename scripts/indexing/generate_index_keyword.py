import json
import logging
import os
import shutil
import time

from govscape.config import DataModel
from govscape.data_loader import RemoteDirectoryIterator, build_data_loader
from govscape.utils import base_argument_parser

import govscape as gs

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


if __name__ == "__main__":
    # FIELDS TO SET --------------------------------------------------------
    parser = base_argument_parser(description="Generate keyword index")
    parser.set_defaults(batch_size=100000)
    parser.add_argument(
        "--keyword_index_type",
        type=str,
        default="LanceDB",
        help="Type of keyword index to use: LanceDB, SQLite or Whoosh",
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

    local_dm = DataModel(LOCAL_DATA_DIR)
    remote_dm = DataModel(REMOTE_DATA_DIR)
    REMOTE_CHECKPOINT_PATH = os.path.join(
        remote_dm.checkpoints_directory, "checkpoint_text_index.json"
    )
    LOCAL_CHECKPOINT_PATH = os.path.join(
        local_dm.checkpoints_directory, "checkpoint_text_index.json"
    )
    REMOTE_PERFORMANCE_PATH = os.path.join(
        remote_dm.performance_directory, "performance_keyword_index.json"
    )
    LOCAL_PERFORMANCE_PATH = os.path.join(
        local_dm.performance_directory, "performance_keyword_index.json"
    )

    if os.path.isdir(LOCAL_DATA_DIR):
        shutil.rmtree(LOCAL_DATA_DIR, ignore_errors=True)

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(local_dm.txt_directory, exist_ok=True)
    os.makedirs(local_dm.index_keyword_directory, exist_ok=True)
    os.makedirs(local_dm.checkpoints_directory, exist_ok=True)
    os.makedirs(local_dm.performance_directory, exist_ok=True)

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
        remote_dm.txt_directory,
        remote_checkpoint_path=REMOTE_CHECKPOINT_PATH,
        local_checkpoint_path=LOCAL_CHECKPOINT_PATH,
        local_dir=local_dm.txt_directory,
    )

    remote_existing_idx_files = data_loader.list_objects(
        remote_dm.index_keyword_directory
    )
    for remote_file in remote_existing_idx_files.keys:
        data_loader.download_file(
            remote_file,
            os.path.join(
                local_dm.index_keyword_directory, os.path.basename(remote_file)
            ),
        )

    # uploads dir of files to backend
    def upload_directory_to_backend(local_dir, remote_dir):
        data_loader.upload_directory(local_dir, remote_dir)

    # process txts: update index and upload to backend
    def process_txt_files(txt_files):
        time_index_start = time.time()
        if INDEX_TYPE == "LanceDB":
            index = gs.LanceDBKeywordIndex(local_dm.index_keyword_directory)
        elif INDEX_TYPE == "SQLite":
            index = gs.SQLiteKeywordIndex(local_dm.index_keyword_directory)
        elif INDEX_TYPE == "Whoosh":
            index = gs.WhooshKeywordIndex(local_dm.index_keyword_directory)
        elif INDEX_TYPE == "Lucene":
            index = gs.LuceneKeywordIndex(local_dm.index_keyword_directory)
        else:
            raise ValueError(
                "index_type must be either 'LanceDB', 'SQLite', 'Whoosh', or 'Lucene'"
            )

        index.load_index()
        names = []
        pages = []
        txts = []
        for txt_file in txt_files:
            txt_file_path = os.path.join(local_dm.txt_directory, txt_file)
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
        data_loader.upload_directory(
            local_dm.index_keyword_directory, remote_dm.index_keyword_directory
        )
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
        # Each page of results corresponds to 1 gzipped batch file
        max_files_to_process = NUM_PAGES_TO_PROCESS
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
                os.path.relpath(path, local_dm.txt_directory).replace("\\", "/")
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
            if os.path.exists(local_dm.txt_directory):
                shutil.rmtree(local_dm.txt_directory)
                os.makedirs(local_dm.txt_directory, exist_ok=True)

        # After all batches are processed, clean up the directories
        if os.path.exists(local_dm.txt_directory):
            shutil.rmtree(local_dm.txt_directory)
        if os.path.exists(local_dm.index_keyword_directory):
            shutil.rmtree(local_dm.index_keyword_directory)

        overall_end_time = time.time()
        print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))

    def main():
        batched_file_download(BATCH_SIZE)

    main()
