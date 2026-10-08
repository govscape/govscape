"""Download a small set of test PDFs and their CDX rows for local development."""

import argparse
import itertools
import logging
import os
import random
from collections.abc import Iterator

import duckdb
import pandas as pd

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
DIGEST_LENGTH = 32


def iter_cdx_row_groups(
    connection: duckdb.DuckDBPyConnection, cdx_url: str, url_filter: str | None
) -> Iterator[pd.DataFrame]:
    """Yield the matching rows of each CDX row group, in random order.

    The full CDX is ~11.5GB and not sorted by digest, so rather than download
    or scan it, whole row groups are read one at a time (DuckDB only fetches
    the row group covering the requested file_row_number range) until the
    caller has enough PDFs. Crawls of a sampled PDF that live in unread row
    groups are missed, which is acceptable for local development.
    """
    row_group_sizes = [
        num_rows
        for (num_rows,) in connection.execute(
            "SELECT row_group_num_rows FROM parquet_metadata(?) "
            "WHERE column_id = 0 ORDER BY row_group_id",
            [cdx_url],
        ).fetchall()
    ]
    row_group_starts = list(itertools.accumulate(row_group_sizes, initial=0))

    query = (
        "SELECT * EXCLUDE (file_row_number) "
        "FROM read_parquet(?, file_row_number = true) "
        "WHERE file_row_number >= ? AND file_row_number < ? AND length(digest) = ?"
    )
    filter_params = []
    if url_filter:
        query += " AND lower(url) LIKE ?"
        filter_params.append(f"%{url_filter.lower()}%")

    row_groups = list(range(len(row_group_sizes)))
    random.shuffle(row_groups)
    for num_read, row_group in enumerate(row_groups, start=1):
        rows = connection.execute(
            query,
            [
                cdx_url,
                row_group_starts[row_group],
                row_group_starts[row_group + 1],
                DIGEST_LENGTH,
                *filter_params,
            ],
        ).df()
        logging.info(
            "Read %d/%d CDX row groups (%d matching rows)",
            num_read,
            len(row_groups),
            len(rows),
        )
        yield rows


def main():
    parser = argparse.ArgumentParser(
        description="Download test PDFs and their CDX rows for local development."
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

    # Download PDFs for random digests from each sampled row group until there
    # are num_pdfs, skipping any digest with no corresponding object in the PDF
    # directory. Only the CDX rows of downloaded PDFs are kept.
    local_pdf_dir = os.path.join(args.local_base_dir, "pdfs")
    os.makedirs(local_pdf_dir, exist_ok=True)
    logging.info("Downloading up to %d PDFs from %s", args.num_pdfs, REMOTE_PDF_DIR)
    downloaded: list[str] = []
    cdx_frames = []
    for cdx_rows in iter_cdx_row_groups(
        duckdb.connect(), data_loader.to_uri(REMOTE_CDX_PATH), args.url_filter
    ):
        candidates = list(set(cdx_rows["digest"]) - set(downloaded))
        random.shuffle(candidates)
        for digest in candidates:
            if len(downloaded) >= args.num_pdfs:
                break
            remote_key = f"{REMOTE_PDF_DIR}{digest}.pdf"
            if not data_loader.exists(remote_key):
                logging.warning(
                    f"Warning: PDF for digest {remote_key} not found in bucket, "
                    "skipping"
                )
                continue
            data_loader.download_file(
                remote_key, os.path.join(local_pdf_dir, f"{digest}.pdf")
            )
            downloaded.append(digest)
        cdx_frames.append(cdx_rows[cdx_rows["digest"].isin(downloaded)])
        if len(downloaded) >= args.num_pdfs:
            break
    logging.info(
        "PDF download complete: %d files in %s", len(downloaded), local_pdf_dir
    )
    if not downloaded:
        return

    # Write CDX rows for every local PDF, keeping rows from earlier runs so the
    # CDX keeps covering PDFs that were already downloaded.
    local_cdx_path = os.path.join(args.local_base_dir, "cdx", "complete_cdx.parquet")
    os.makedirs(os.path.dirname(local_cdx_path), exist_ok=True)
    if os.path.exists(local_cdx_path):
        local_digests = [
            name.removesuffix(".pdf")
            for name in os.listdir(local_pdf_dir)
            if name.endswith(".pdf")
        ]
        cdx_frames.append(
            pd.read_parquet(local_cdx_path, filters=[("digest", "in", local_digests)])
        )
    cdx = pd.concat(cdx_frames, ignore_index=True).drop_duplicates()
    cdx.to_parquet(local_cdx_path, index=False)
    logging.info(
        "Wrote %d CDX rows for %d PDFs to %s",
        len(cdx),
        cdx["digest"].nunique(),
        local_cdx_path,
    )


main()
