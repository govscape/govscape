import argparse
import os
import re
from urllib.parse import urlparse

_WORD_PATTERN = re.compile(r"\w+")


def read_txt_file(txt_path):
    with open(txt_path) as file:
        return file.read()


def load_word_blacklist(path: str) -> set[str]:
    """Load a newline-delimited word blacklist file into a lowercased set.

    Blank lines and lines starting with "#" are ignored, matching the
    format used by the PDF digest blacklist.txt (see DATA_MODEL.md).
    """
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


def contains_blacklisted_word(text: str, blacklist: set[str]) -> bool:
    """Return True if `text` contains any whole word from `blacklist`.

    Runs in O(n) time (n = len(text)) regardless of blacklist size: words
    are tokenized with a single regex pass and looked up in a hash set
    (O(1) per word), and the scan stops at the first match. `blacklist`
    entries must already be lowercased (e.g. via `load_word_blacklist`).
    """
    if not blacklist:
        return False
    return any(
        match.group() in blacklist for match in _WORD_PATTERN.finditer(text.lower())
    )


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

    Includes: --backend, --bucket_name, --local_base_dir,
    --num_pages_to_process, --batch_size, --remote_data_dir.
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
    parser.add_argument("--bucket_name", type=str, help="S3 Bucket Name")
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
