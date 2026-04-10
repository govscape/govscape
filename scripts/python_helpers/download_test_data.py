"""Download a small set of test PDFs and the CDX parquet for local development."""

import argparse
import logging
import os

from govscape.data_loader import RemoteDirectoryIterator, build_data_loader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

REMOTE_PDF_DIR = "archive-small/PDFs/"
REMOTE_CDX_PATH = "archive/2020/CDX/pdf_metadata.parquet"


def main():
    parser = argparse.ArgumentParser(
        description="Download test PDFs and CDX parquet for local development."
    )
    parser.add_argument("--bucket_name", required=True, help="S3 bucket name")
    parser.add_argument(
        "--local_base_dir",
        default="data/s3_mock",
        help="Local directory mirroring the S3 bucket structure",
    )
    parser.add_argument(
        "--num_pdfs",
        type=int,
        default=1000,
        help="Maximum number of PDFs to download",
    )
    args = parser.parse_args()

    data_loader = build_data_loader(
        "s3",
        args.bucket_name,
        local_base_dir=args.local_base_dir,
    )

    # Download CDX parquet
    local_cdx_path = os.path.join(
        args.local_base_dir, "archive", "CDX", "pdf_metadata.parquet"
    )
    os.makedirs(os.path.dirname(local_cdx_path), exist_ok=True)
    logging.info("Downloading CDX parquet to %s", local_cdx_path)
    data_loader.download_file(REMOTE_CDX_PATH, local_cdx_path)
    logging.info("CDX download complete")

    # Download PDFs via RemoteDirectoryIterator (stops after num_pdfs keys)
    local_pdf_dir = os.path.join(args.local_base_dir, "archive", "PDFs")
    local_checkpoint_path = os.path.join(
        args.local_base_dir, "checkpoints", "checkpoint_test_data.json"
    )
    remote_checkpoint_path = "checkpoints/checkpoint_test_data.json"
    os.makedirs(local_pdf_dir, exist_ok=True)
    os.makedirs(os.path.dirname(local_checkpoint_path), exist_ok=True)
    logging.info("Downloading up to %d PDFs from %s", args.num_pdfs, REMOTE_PDF_DIR)
    pdf_iter = RemoteDirectoryIterator(
        data_loader,
        REMOTE_PDF_DIR,
        remote_checkpoint_path=remote_checkpoint_path,
        local_checkpoint_path=local_checkpoint_path,
        local_dir=local_pdf_dir,
    )
    local_paths = pdf_iter.download_batch(
        max_keys=args.num_pdfs,
        filter_fn=lambda key: key.endswith(".pdf"),
    )
    logging.info(
        "PDF download complete: %d files in %s", len(local_paths), local_pdf_dir
    )


main()
