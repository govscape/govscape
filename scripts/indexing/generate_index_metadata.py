import json
import logging
import os
import shutil
import time

import duckdb
from govscape.data_loader import RemoteDirectoryIterator, build_data_loader
from govscape.indexing import SQLiteMetadataIndex
from govscape.utils import base_argument_parser, extract_subdomain

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


BATCH_SIZE = 100000


def main():
    parser = base_argument_parser(description="Generate metadata index")
    parser.add_argument(
        "--cdx_parquet_key", required=True, help="S3 Key for CDX parquet file"
    )
    args = parser.parse_args()
    BUCKET_NAME = args.bucket_name
    NUM_PAGES_TO_PROCESS = args.num_pages_to_process

    # ---------------------------------------------------------------------------
    LOCAL_DATA_DIR = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")),
        "data",
        "prod",
    )
    REMOTE_DATA_DIR = args.remote_data_dir
    REMOTE_CDX_PATH = args.cdx_parquet_key
    REMOTE_METADATA_DIR = os.path.join(REMOTE_DATA_DIR, "metadata")
    LOCAL_METADATA_DIR = os.path.join(LOCAL_DATA_DIR, "metadata")
    LOCAL_CDX_PATH = os.path.join(LOCAL_DATA_DIR, "CDX", "cdx_metadata.parquet")
    LOCAL_INDEX_PATH = os.path.join(LOCAL_DATA_DIR, "metadata.db")
    REMOTE_INDEX_PATH = os.path.join(REMOTE_DATA_DIR, "index_metadata", "metadata.db")
    REMOTE_CHECKPOINT_PATH = os.path.join(
        REMOTE_DATA_DIR, "checkpoints", "checkpoint_metadata.json"
    )
    LOCAL_CHECKPOINT_PATH = os.path.join(
        LOCAL_DATA_DIR, "checkpoints", "checkpoint_metadata.json"
    )

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOCAL_DATA_DIR, "CDX"), exist_ok=True)
    os.makedirs(LOCAL_METADATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOCAL_CHECKPOINT_PATH), exist_ok=True)

    # ---------------------------------------------------------------------------
    data_loader = build_data_loader(
        args.backend,
        BUCKET_NAME,
        local_base_dir=args.local_base_dir,
    )

    remote_iter = RemoteDirectoryIterator(
        data_loader,
        REMOTE_METADATA_DIR,
        remote_checkpoint_path=REMOTE_CHECKPOINT_PATH,
        local_checkpoint_path=LOCAL_CHECKPOINT_PATH,
        local_dir=LOCAL_METADATA_DIR,
    )

    # Download the CDX parquet file from S3
    print("Reading CDX data")
    os.makedirs(os.path.dirname(LOCAL_CDX_PATH), exist_ok=True)
    if not os.path.exists(LOCAL_CDX_PATH):
        data_loader.download_file(REMOTE_CDX_PATH, LOCAL_CDX_PATH)

    print("Initializing Index")

    # Initialize the SQLite metadata index
    try:
        data_loader.download_file(REMOTE_INDEX_PATH, LOCAL_INDEX_PATH)
    except Exception as e:
        print(f"No Existing Index File Found: {e}")
    index = SQLiteMetadataIndex(LOCAL_DATA_DIR)

    # Create the metadata table
    index.build_index()
    files_processed = 0

    # Each page of results corresponds to 1 gzipped batch file
    max_files_to_process = NUM_PAGES_TO_PROCESS
    # get the metadata files from backend
    while files_processed < max_files_to_process:
        os.makedirs(LOCAL_METADATA_DIR, exist_ok=True)
        start_time = time.time()
        print("Downloading Metadata Files from backend")
        batch_limit = min(BATCH_SIZE, max_files_to_process - files_processed)
        local_paths = remote_iter.download_batch(
            max_keys=batch_limit,
            filter_fn=lambda key: key.endswith(".json"),
        )
        if not local_paths:
            break

        print("Adding Metadata to Index")
        rows = []
        for filepath in local_paths:
            row = _read_metadata_row(filepath)
            if row is not None:
                rows.append(row)

        con = duckdb.connect()
        con.execute("CREATE TEMP TABLE rows_rel (digest VARCHAR, num_pages BIGINT)")
        con.executemany(
            "INSERT INTO rows_rel VALUES (?, ?)",
            [(r["digest"], r["num_pages"]) for r in rows],
        )
        result = con.execute(
            """
            SELECT r.digest, r.num_pages, c.url, c.crawl_date
            FROM rows_rel r
            INNER JOIN (
                SELECT REPLACE(digest, 'sha1:', '') AS digest, url, crawl_date
                FROM read_parquet(?)
            ) c ON r.digest = c.digest
            """,
            [LOCAL_CDX_PATH],
        ).fetchall()
        con.close()

        print("Building Index")
        index.build_index()
        cur_batch = []
        rows_added = 0
        for digest, num_pages, url, crawl_date in result:
            cur_batch.append(
                {
                    "crawl_url": url,
                    "crawl_date": crawl_date,
                    "pdf_name": digest,
                    "sub_domain": extract_subdomain(url),
                    "page_count": num_pages,
                    "s3_url": data_loader.to_uri(
                        os.path.join("archive/2020/PDFs", f"{digest}.pdf")
                    ),
                }
            )
            if len(cur_batch) >= 1000:
                index.add_batch(cur_batch)
                rows_added += len(cur_batch)
                print(f"Added {rows_added} rows to index")
                cur_batch = []
        if len(cur_batch) > 0:
            index.add_batch(cur_batch)
            rows_added += len(cur_batch)
            print(f"Added {rows_added} rows to index")

        print("Uploading Index")
        data_loader.upload_file(LOCAL_INDEX_PATH, REMOTE_INDEX_PATH)

        remote_iter.save_checkpoint()
        files_processed += len(local_paths)
        print(
            "Files Processed", files_processed, "Total Time:", time.time() - start_time
        )
        try:
            if os.path.exists(LOCAL_METADATA_DIR):
                shutil.rmtree(LOCAL_METADATA_DIR)
        except Exception as e:
            print(f"Failed to remove {LOCAL_METADATA_DIR}: {e}")
    print("Saving Index")
    index.save_index()

    print("Uploading Index")
    data_loader.upload_file(LOCAL_INDEX_PATH, REMOTE_INDEX_PATH)


def _read_metadata_row(filepath):
    try:
        with open(filepath) as f:
            metadata_json = json.load(f)
        digest_val = os.path.dirname(filepath).split("/")[-1]
        os.remove(filepath)  # Clean up the file after reading
        return {
            "digest": digest_val,
            "num_pages": metadata_json.get("num_pages", None),
        }
    except Exception as exc:
        print(f"Error reading {filepath}: {exc}")
        return None


main()
