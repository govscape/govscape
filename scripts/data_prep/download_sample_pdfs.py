"""Download a small set of test PDFs and the CDX parquet for local development."""

import argparse
import logging
import os

import duckdb

from govscape.config import PDF_ARCHIVE_BUCKET, PDF_ARCHIVE_CDX_KEY, PDF_ARCHIVE_PDF_DIR
from govscape.data_loader import build_data_loader

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)

REMOTE_PDF_DIR = PDF_ARCHIVE_PDF_DIR
REMOTE_CDX_PATH = PDF_ARCHIVE_CDX_KEY


def main():
    parser = argparse.ArgumentParser(
        description="Download test PDFs and CDX parquet for local development."
    )
    parser.add_argument(
        "--bucket_name",
        default=PDF_ARCHIVE_BUCKET,
        help="S3 bucket (with optional key prefix) holding the PDF archive",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="AWS credentials profile (default: AWS_PROFILE, else anonymous)",
    )
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
    parser.add_argument(
        "--url_filter",
        default=None,
        help="Only download PDFs whose source URL contains this substring "
        "(case-insensitive). Omit to download all PDFs.",
    )
    args = parser.parse_args()

    data_loader = build_data_loader(
        "s3",
        args.bucket_name,
        local_base_dir=args.local_base_dir,
        profile_name=args.profile,
    )

    # Download CDX parquet
    local_cdx_path = os.path.join(args.local_base_dir, "cdx", "complete_cdx.parquet")
    os.makedirs(os.path.dirname(local_cdx_path), exist_ok=True)
    if not os.path.exists(local_cdx_path):
        logging.info("Downloading CDX parquet to %s", local_cdx_path)
        data_loader.download_file(REMOTE_CDX_PATH, local_cdx_path)
        logging.info("CDX download complete")

    # Determine which PDFs to download by querying the CDX for matching digests.
    # Order randomly so we download a random sample rather than the first
    # num_pdfs digests in CDX order.
    query = "SELECT DISTINCT digest FROM read_parquet(?)"
    params = [local_cdx_path]
    if args.url_filter:
        query += " WHERE lower(url) LIKE ?"
        params.append(f"%{args.url_filter.lower()}%")
    query += " ORDER BY random()"
    digests = [row[0] for row in duckdb.connect().execute(query, params).fetchall()]
    logging.info("CDX query matched %d candidate digests", len(digests))

    # Download matching PDFs directly, up to num_pdfs, skipping any digest with no
    # corresponding object in the PDF directory.
    local_pdf_dir = os.path.join(args.local_base_dir, "pdfs")
    os.makedirs(local_pdf_dir, exist_ok=True)
    logging.info("Downloading up to %d PDFs from %s", args.num_pdfs, REMOTE_PDF_DIR)
    local_paths = []
    for digest in digests:
        if len(local_paths) >= args.num_pdfs:
            break
        remote_key = f"{REMOTE_PDF_DIR}{digest}.pdf"
        if not data_loader.exists(remote_key):
            logging.warning(
                f"Warning: PDF for digest {remote_key} not found in bucket, skipping"
            )
            continue
        local_path = os.path.join(local_pdf_dir, f"{digest}.pdf")
        data_loader.download_file(remote_key, local_path)
        local_paths.append(local_path)
    logging.info(
        "PDF download complete: %d files in %s", len(local_paths), local_pdf_dir
    )


main()
