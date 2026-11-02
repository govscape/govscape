import argparse
import itertools
import logging
import random
from collections.abc import Iterator
from urllib.parse import urlparse

import duckdb
import pandas as pd

from govscape.config import (
    DERIVATIVE_BUCKET,
    PDF_ARCHIVE_BUCKET,
    SOURCE_COOP_ENDPOINT,
)

DIGEST_LENGTH = 32


def read_txt_file(txt_path):
    with open(txt_path) as file:
        return file.read()


def str2bool(v):
    """Argparse type for boolean flags that accepts yes/no/true/false/1/0."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def extract_subdomain(url):
    """Extract the base domain (e.g. 'example.gov') from a full URL."""
    parsed = urlparse(url)
    hostname = parsed.hostname
    if hostname is None:
        return None
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def endpoint_url_arg(value: str) -> str | None:
    """Argparse type for S3 endpoints; an empty string selects AWS S3."""
    return value or None


def base_argument_parser(description="GovScape script"):
    """Return an ArgumentParser pre-loaded with common flags.

    Includes: --backend, --bucket_name, --pdf_bucket_name, --profile,
    --pdf_profile, --endpoint_url, --local_base_dir, --num_pages_to_process,
    --batch_size, --remote_data_dir.
    --bucket_name is the store for derived data and --pdf_bucket_name is the
    store for the raw PDFs and CDX. --profile/--pdf_profile select the AWS
    credentials profile used for each, and --endpoint_url the S3 endpoint for
    both (an empty value means AWS S3).
    Callers can add more arguments or override defaults via
    ``parser.set_defaults()``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--backend",
        choices=["s3", "local"],
        default="s3",
        help="Data backend to use",
    )
    parser.add_argument(
        "--bucket_name",
        type=str,
        default=DERIVATIVE_BUCKET,
        help="S3 bucket (with optional key prefix) holding derived data",
    )
    parser.add_argument(
        "--pdf_bucket_name",
        type=str,
        default=PDF_ARCHIVE_BUCKET,
        help="S3 bucket (with optional key prefix) holding the PDF archive",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="AWS credentials profile for --bucket_name (default: AWS_PROFILE)",
    )
    parser.add_argument(
        "--pdf_profile",
        type=str,
        default=None,
        help="AWS credentials profile for --pdf_bucket_name (default: AWS_PROFILE)",
    )
    parser.add_argument(
        "--endpoint_url",
        type=endpoint_url_arg,
        default=SOURCE_COOP_ENDPOINT,
        help="S3 endpoint (default: the source.coop proxy; '' for AWS S3)",
    )
    parser.add_argument(
        "--local_base_dir",
        type=str,
        default="data",
        help="Base directory for local backend",
    )
    parser.add_argument(
        "--num_pages_to_process",
        type=int,
        default=100,
        help="Number of pages to process",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of items to process at a time",
    )
    parser.add_argument(
        "--remote_data_dir",
        type=str,
        help="Remote data directory",
    )
    return parser


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
