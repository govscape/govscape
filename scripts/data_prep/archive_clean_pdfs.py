import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from govscape.data_loader import build_data_loader
from govscape.utils import base_argument_parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _copy_digest(data_loader, dirty_prefix, clean_prefix, digest):
    try:
        clean_digest = digest.replace("sha1:", "")
        clean_file = os.path.join(clean_prefix, clean_digest + ".pdf")
        dirty_file = os.path.join(dirty_prefix, digest + ".pdf")
        data_loader.copy_object(dirty_file, clean_file)
        return True
    except Exception:
        return False


def copy_to_clean(backend, bucket, local_base_dir, dirty_prefix, clean_prefix, digests):
    data_loader = build_data_loader(backend, bucket, local_base_dir)
    copied_properly = 0
    for digest in digests:
        if _copy_digest(data_loader, dirty_prefix, clean_prefix, digest):
            copied_properly += 1

    print(f"Copied {copied_properly} files to clean directory.")
    return digests


def _safe_future_result(future):
    try:
        return future.result()
    except Exception as exc:
        print(f"Error downloading {future}: {exc}")
        return []


if __name__ == "__main__":
    # FIELDS TO SET --------------------------------------------------------
    parser = base_argument_parser(description="Archive clean PDFs")
    parser.set_defaults(num_pages_to_process=1000000, batch_size=1000000)
    parser.add_argument(
        "--metadata_prefix", type=str, help="S3 Directory for metadata data"
    )
    parser.add_argument(
        "--clean_data_prefix", type=str, help="S3 Directory for input data"
    )
    parser.add_argument(
        "--dirty_data_prefix", type=str, help="S3 Directory for output data"
    )
    args = parser.parse_args()

    NUM_PAGES_TO_PROCESS = args.num_pages_to_process
    BATCH_SIZE = args.batch_size

    bucket_name = args.bucket_name  # 'bcgl-public-bucket'
    metadata_prefix = args.metadata_prefix  # 'prod-serving/'# INPUT DATA DIR IN S3 HERE
    clean_data_prefix = (
        args.clean_data_prefix
    )  # 'prod-serving/'# INPUT DATA DIR IN S3 HERE
    dirty_data_prefix = (
        args.dirty_data_prefix
    )  # 'prod-serving/' # OUTPUT OVERALL DATA DIR IN S3 HERE

    # ---------------------------------------------------------------------------
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "prod")

    # Token to track of which pages have already been processed.
    progress_path = "clean_copy_progress.json"

    data_loader = build_data_loader(args.backend, bucket_name, args.local_base_dir)

    # gets txt files from backend
    def list_digests(num_pages=1):
        digests = []
        pdfs_retrieved = 0
        while True:
            result = data_loader.list_objects(metadata_prefix)
            contents = result.keys
            digest_batch = [
                os.path.split(os.path.split(key)[0])[1]
                for key in contents
                if key.endswith(".json")
            ]
            digests.extend(digest_batch)
            pdfs_retrieved += 1
            # with open(progress_path, "w") as f:
            #     json.dump({"continuation_token": continuation_token}, f)
            print(contents)
            if not result.is_truncated:
                return digests, False
            if pdfs_retrieved >= num_pages or not result.is_truncated:
                break
        print(f"Listed {len(contents)} files from backend")
        return digests, True

    # overall method that gets the files in batches and runs them through the pipeline
    def batched_file_download(BATCH_SIZE):
        # result = s3.list_objects_v2(Bucket=bucket_name, Prefix=pdfs_dir)
        # # get list of pdf file names
        # pdf_files = [obj["Key"] for obj in result.get("Contents", [])
        #              if obj["Key"].endswith(".pdf")]  # note this only returns 1000

        overall_start_time = time.time()
        for _i in range(1, math.ceil(NUM_PAGES_TO_PROCESS / 1000)):
            # get the pdf files from s3
            digests, is_finished = list_digests(1000)
            print("Now starting with total number of PDF files: ", len(digests))

            for j in range(0, len(digests), BATCH_SIZE):
                print("-" * 93)
                print("WE ARE ON BATCH: ", j)
                print("-" * 93)
                batch = digests[j : j + BATCH_SIZE]
                num_workers = 512
                worker_batches = np.array_split(
                    batch, num_workers
                )  # Split the batch into smaller batches for parallel downloading
                local_batch = []
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(
                            copy_to_clean,
                            args.backend,
                            bucket_name,
                            args.local_base_dir,
                            dirty_data_prefix,
                            clean_data_prefix,
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
            print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))

    def main():
        batched_file_download(BATCH_SIZE)

    main()
