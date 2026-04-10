import argparse
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from govscape.data_loader import build_data_loader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
# to run this file: poetry run python s3_ec2_embedding_pipeline.py
# ---------------------------------------------------------------------------


def _copy_and_delete(data_loader, sha1_key):
    clean_key = sha1_key.replace("sha1:", "")
    if clean_key == sha1_key:
        return False, False
    try:
        data_loader.copy_object(sha1_key, clean_key)
        copied = True
    except Exception:
        return False, False
    try:
        data_loader.delete_object(sha1_key)
        deleted = True
    except Exception as exc:
        print(f"Error deleting {sha1_key}: {exc}")
        deleted = False
    return copied, deleted


def remove_sha1(backend, bucket, local_base_dir, sha1_keys):
    data_loader = build_data_loader(backend, bucket, local_base_dir)
    copied_properly = 0
    deleted_properly = 0
    for sha1_key in sha1_keys:
        copied, deleted = _copy_and_delete(data_loader, sha1_key)
        if copied:
            copied_properly += 1
        if deleted:
            deleted_properly += 1

    print(f"Copied {copied_properly} files to clean key.")
    print(f"Deleted {deleted_properly} files from directory.")
    return sha1_keys


def _safe_future_result(future):
    try:
        return future.result()
    except Exception as exc:
        print(f"Error downloading {future}: {exc}")
        return []


if __name__ == "__main__":
    # FIELDS TO SET --------------------------------------------------------
    parser = argparse.ArgumentParser(description="S3 EC2 Embedding Pipeline")
    parser.add_argument(
        "--num_pages_to_process",
        type=int,
        default=1000000,
        help="Number of pages to process from S3",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000000,
        help="Number of pages to process at a time",
    )
    parser.add_argument("--bucket_name", type=str, help="S3 Bucket Name")
    parser.add_argument(
        "--data_prefix", type=str, help="S3 Directory to be sha1 cleaned"
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

    bucket_name = args.bucket_name  # 'bcgl-public-bucket'
    data_prefix = args.data_prefix  # 'prod-serving/'# INPUT DATA DIR IN S3 HERE

    # ---------------------------------------------------------------------------
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "prod")

    txt_directory = os.path.join(DATA_DIR, "txt")
    index_keyword_directory = os.path.join(DATA_DIR, "index_keyword")

    # Token to track of which pages have already been processed.
    progress_path = "sha1_clean_progress.json"

    data_loader = build_data_loader(args.backend, bucket_name, args.local_base_dir)

    # gets txt files from backend
    def list_digests(num_pages=1):
        keys = []
        keys_retrieved = 0
        while True:
            result = data_loader.list_objects(data_prefix)

            key_batch = result.keys
            keys.extend(key_batch)
            keys_retrieved += 1
            if not result.is_truncated:
                return keys, True
            if keys_retrieved >= num_pages:
                break
        print(f"Listed {len(result.keys)} files from backend")
        return keys, False

    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE):
        batch_size = 100
        overall_start_time = time.time()
        for _i in range(math.ceil(NUM_PAGES_TO_PROCESS / batch_size)):
            # get the pdf files from s3
            digests, is_finished = list_digests(batch_size)
            print("Now starting with total number of PDF files: ", len(digests))

            for j in range(0, len(digests), BATCH_SIZE):
                print("-" * 93)
                print("WE ARE ON BATCH: ", j)
                print("-" * 93)
                batch = digests[j : j + BATCH_SIZE]
                num_workers = 64
                worker_batches = np.array_split(
                    batch, num_workers
                )  # Split the batch into smaller batches for parallel downloading
                local_batch = []
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            remove_sha1,
                            args.backend,
                            bucket_name,
                            args.local_base_dir,
                            worker_batch,
                        )
                        for worker_batch in worker_batches
                    ]
                    for future in as_completed(futures):
                        local_batch.extend(_safe_future_result(future))
                data_loader.save_checkpoint()
            if is_finished:
                break
            overall_end_time = time.time()
            print("TOTAL TIME TO RENAME IS ", (overall_end_time - overall_start_time))

    def main():
        batched_file_download(BATCH_SIZE)

    main()
