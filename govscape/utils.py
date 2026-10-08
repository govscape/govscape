import argparse
from urllib.parse import urlparse

from govscape.config import (
    DERIVATIVE_BUCKET,
    PDF_ARCHIVE_BUCKET,
)


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


def base_argument_parser(description="GovScape script"):
    """Return an ArgumentParser pre-loaded with common flags.

    Includes: --backend, --bucket_name, --pdf_bucket_name, --profile,
    --pdf_profile, --local_base_dir, --num_pages_to_process, --batch_size,
    --remote_data_dir.
    --bucket_name is the store for derived data and --pdf_bucket_name is the
    store for the raw PDFs and CDX. --profile/--pdf_profile select the AWS
    credentials profile used for each.
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
