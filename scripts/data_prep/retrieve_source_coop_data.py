import argparse
import logging
import random
import tempfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import requests

from govscape.config import (
    PDF_ARCHIVE_BUCKET,
    PDF_ARCHIVE_CDX_KEY,
    PDF_ARCHIVE_PDF_DIR,
    SOURCE_COOP_ENDPOINT,
)
from govscape.utils import iter_cdx_row_groups

ARCHIVE_URL = f"{SOURCE_COOP_ENDPOINT}/{PDF_ARCHIVE_BUCKET}"
PARQUET_URL = f"{ARCHIVE_URL}{PDF_ARCHIVE_CDX_KEY}"
PDF_BASE_URL = f"{ARCHIVE_URL}{PDF_ARCHIVE_PDF_DIR}".rstrip("/")
DEFAULT_PDF_DIR = Path("tests/test_data/pdfs")
DEFAULT_CDX_DIR = Path("tests/test_data/cdx")
CDX_SAMPLE_FILENAME = "complete_cdx_sample.parquet"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def _download_pdf(session: requests.Session, digest: str, pdf_dir: Path) -> Path:
    pdf_path = pdf_dir / f"{digest}.pdf"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=pdf_dir, prefix=f".{digest}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with session.get(
                f"{PDF_BASE_URL}/{digest}.pdf", stream=True, timeout=(10, 120)
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        temporary_file.write(chunk)
        temporary_path.replace(pdf_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return pdf_path


def _save_cdx_rows(
    connection: duckdb.DuckDBPyConnection,
    digests: list[str],
    cdx_path: Path,
    known_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write the CDX rows of the given digests to cdx_path and return them.

    Rows in known_rows (already read from the CDX) are reused; only the
    remaining digests are looked up in the remote CDX, which requires scanning
    its whole digest column because the CDX is not sorted by digest.
    """
    if not digests:
        raise ValueError("No PDF digests found; cannot create a matching CDX sample.")

    frames = []
    lookup_digests = list(digests)
    if known_rows is not None and not known_rows.empty:
        known_rows = known_rows[known_rows["digest"].isin(digests)]
        frames.append(known_rows)
        lookup_digests = sorted(set(digests) - set(known_rows["digest"]))
    if lookup_digests:
        logger.info(
            "Looking up CDX rows for %d PDFs in the remote CDX", len(lookup_digests)
        )
        placeholders = ", ".join("?" for _ in lookup_digests)
        frames.append(
            connection.execute(
                f"SELECT * FROM read_parquet(?) WHERE digest IN ({placeholders})",
                [PARQUET_URL, *lookup_digests],
            ).df()
        )
    rows = pd.concat(frames, ignore_index=True).drop_duplicates()
    missing_digests = set(digests) - set(rows["digest"])
    if missing_digests:
        logger.warning(
            "Skipping PDFs with no CDX records: %s", ", ".join(sorted(missing_digests))
        )
    if rows.empty:
        raise ValueError("No CDX records found for any PDF digest.")
    rows.to_parquet(cdx_path, index=False)
    return rows


def _iter_candidate_rows(
    connection: duckdb.DuckDBPyConnection, url_filter: str | None, sample: bool
) -> Iterator[pd.DataFrame]:
    """Yield batches of CDX rows to pick PDFs from.

    By default rows are streamed in CDX order, so DuckDB stops reading the remote
    parquet once enough PDFs have been downloaded. With sample=True, whole row
    groups are read in random order instead (see iter_cdx_row_groups).
    """
    if sample:
        yield from iter_cdx_row_groups(connection, PARQUET_URL, url_filter)
        return

    query = "SELECT * FROM read_parquet(?) WHERE digest IS NOT NULL AND digest <> ''"
    params = [PARQUET_URL]
    if url_filter:
        query += " AND lower(url) LIKE ?"
        params.append(f"%{url_filter.lower()}%")
    result = connection.execute(query, params)
    while not (rows := result.fetch_df_chunk()).empty:
        yield rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download archived PDFs and their CDX rows from source.coop."
    )
    parser.add_argument(
        "--num_pdfs",
        type=int,
        default=100,
        help="Number of distinct PDFs to download (default: 100)",
    )
    parser.add_argument(
        "--pdf_dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help=f"Directory for downloaded PDFs (default: {DEFAULT_PDF_DIR})",
    )
    parser.add_argument(
        "--cdx_dir",
        type=Path,
        default=DEFAULT_CDX_DIR,
        help=f"Directory for the matching CDX sample (default: {DEFAULT_CDX_DIR})",
    )
    parser.add_argument(
        "--url_filter",
        default=None,
        help="Only download PDFs whose source URL contains this substring "
        "(case-insensitive). Omit to download all PDFs.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Download a random sample instead of the first digests in CDX order, "
        "drawn from randomly chosen row groups of the remote CDX.",
    )
    args = parser.parse_args()
    if args.num_pdfs <= 0:
        parser.error("--num_pdfs must be greater than zero")

    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.cdx_dir.mkdir(parents=True, exist_ok=True)

    existing_digests = {path.stem for path in args.pdf_dir.glob("*.pdf")}
    if existing_digests:
        logger.info(
            "Skipping %d PDFs already in %s", len(existing_digests), args.pdf_dir
        )

    connection = duckdb.connect()
    seen_digests = set(existing_digests)
    downloaded: list[str] = []
    downloaded_rows = []
    with requests.Session() as session:
        for candidate_rows in _iter_candidate_rows(
            connection, args.url_filter, args.random
        ):
            # dict.fromkeys de-duplicates while keeping CDX order.
            candidates = [
                digest
                for digest in dict.fromkeys(candidate_rows["digest"])
                if digest not in seen_digests
            ]
            if args.random:
                random.shuffle(candidates)
            for digest in candidates:
                if len(downloaded) == args.num_pdfs:
                    break
                seen_digests.add(digest)
                try:
                    _download_pdf(session, digest, args.pdf_dir)
                except requests.RequestException as exc:
                    logger.warning(
                        "Failed to download PDF for digest %s: %s", digest, exc
                    )
                    continue
                downloaded.append(digest)
                logger.info("Downloaded %d of %d PDFs", len(downloaded), args.num_pdfs)
            downloaded_rows.append(
                candidate_rows[candidate_rows["digest"].isin(downloaded)]
            )
            if len(downloaded) == args.num_pdfs:
                break

    # The CDX sample and manifest both describe every PDF in pdf_dir, so they stay
    # consistent across repeated runs into the same directory. CDX rows already
    # read while picking PDFs, or saved by an earlier run, are reused so that the
    # remote CDX is only searched for the remaining PDFs. A new PDF's crawls that
    # appear in parts of the CDX that were not read are therefore not included.
    pdf_digests = sorted(existing_digests | set(downloaded))
    if pdf_digests:
        cdx_path = args.cdx_dir / CDX_SAMPLE_FILENAME
        if cdx_path.exists():
            downloaded_rows.append(
                pd.read_parquet(cdx_path, filters=[("digest", "in", pdf_digests)])
            )
        known_rows = (
            pd.concat(downloaded_rows, ignore_index=True) if downloaded_rows else None
        )
        rows = _save_cdx_rows(connection, pdf_digests, cdx_path, known_rows)
        manifest = rows.groupby("digest", as_index=False)["url"].min()
        manifest.insert(0, "file", manifest["digest"] + ".pdf")
        manifest_path = args.cdx_dir / "digests_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        logger.info("Saved matching CDX rows to %s", cdx_path)
        logger.info("Saved digest manifest to %s", manifest_path)
    connection.close()

    if len(downloaded) != args.num_pdfs:
        raise RuntimeError(
            f"Downloaded {len(downloaded)} of {args.num_pdfs} requested PDFs; "
            "see warnings above for failed downloads."
        )

    logger.info("Downloaded %d PDFs to %s", len(downloaded), args.pdf_dir)


if __name__ == "__main__":
    main()
