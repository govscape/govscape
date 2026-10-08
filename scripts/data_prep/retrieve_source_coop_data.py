import argparse
import logging
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import requests

PARQUET_URL = (
    "https://data.source.coop/govscape/eota-pdf-archive/cdx/complete_cdx.parquet"
)
PDF_BASE_URL = "https://data.source.coop/govscape/eota-pdf-archive/pdfs"
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
    connection: duckdb.DuckDBPyConnection, digests: list[str], cdx_path: Path
) -> pd.DataFrame:
    if not digests:
        raise ValueError("No PDF digests found; cannot create a matching CDX sample.")

    placeholders = ", ".join("?" for _ in digests)
    rows = connection.execute(
        f"SELECT * FROM read_parquet(?) WHERE digest IN ({placeholders})",
        [PARQUET_URL, *digests],
    ).df()
    missing_digests = set(digests) - set(rows["digest"])
    if missing_digests:
        logger.warning(
            "Skipping PDFs with no CDX records: %s", ", ".join(sorted(missing_digests))
        )
    if rows.empty:
        raise ValueError("No CDX records found for any PDF digest.")
    rows.to_parquet(cdx_path, index=False)
    return rows


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
        help="Download a random sample instead of the first digests in CDX order. "
        "This scans the whole remote CDX before the first download.",
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

    # Stream candidates without GROUP BY so DuckDB stops reading the remote parquet
    # once enough new digests have been downloaded (unless --random forces a sort).
    query = (
        "SELECT digest FROM read_parquet(?) WHERE digest IS NOT NULL AND digest <> ''"
    )
    params = [PARQUET_URL]
    if args.url_filter:
        query += " AND lower(url) LIKE ?"
        params.append(f"%{args.url_filter.lower()}%")
    if args.random:
        query += " ORDER BY random()"
    connection = duckdb.connect()
    candidates = connection.execute(query, params)

    seen_digests = set(existing_digests)
    downloaded: list[str] = []
    with requests.Session() as session:
        while len(downloaded) < args.num_pdfs:
            candidate_batch = candidates.fetchmany(1000)
            if not candidate_batch:
                break
            for (digest,) in candidate_batch:
                if digest in seen_digests:
                    continue
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
                if len(downloaded) == args.num_pdfs:
                    break

    # The CDX sample and manifest both describe every PDF in pdf_dir, so they stay
    # consistent across repeated runs into the same directory.
    pdf_digests = sorted(existing_digests | set(downloaded))
    if pdf_digests:
        cdx_path = args.cdx_dir / CDX_SAMPLE_FILENAME
        rows = _save_cdx_rows(connection, pdf_digests, cdx_path)
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
