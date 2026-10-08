import argparse
import logging
import random
import tempfile
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


def _load_cdx_rows(
    connection: duckdb.DuckDBPyConnection, digests: list[str]
) -> pd.DataFrame:
    """Return all CDX rows for the given digests.

    The CDX is not sorted by digest, so this scans the whole remote file; it is
    only used for local PDFs that have no CDX rows saved yet.
    """
    placeholders = ", ".join("?" for _ in digests)
    rows = connection.execute(
        f"SELECT * FROM read_parquet(?) WHERE digest IN ({placeholders})",
        [PARQUET_URL, *digests],
    ).df()
    missing_digests = set(digests) - set(rows["digest"])
    if missing_digests:
        raise ValueError(
            "No CDX records found for PDF digests: "
            + ", ".join(sorted(missing_digests))
        )
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
    args = parser.parse_args()
    if args.num_pdfs <= 0:
        parser.error("--num_pdfs must be greater than zero")

    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    args.cdx_dir.mkdir(parents=True, exist_ok=True)
    cdx_path = args.cdx_dir / CDX_SAMPLE_FILENAME

    # Download PDFs for random digests from randomly sampled CDX row groups,
    # skipping PDFs that are already present.
    connection = duckdb.connect()
    downloaded: list[str] = []
    cdx_frames = []
    with requests.Session() as session:
        for cdx_rows in iter_cdx_row_groups(connection, PARQUET_URL, ".pdf"):
            local_digests = {path.stem for path in args.pdf_dir.glob("*.pdf")}
            candidates = list(set(cdx_rows["digest"]) - local_digests)
            random.shuffle(candidates)
            for digest in candidates:
                if len(downloaded) >= args.num_pdfs:
                    break
                try:
                    _download_pdf(session, digest, args.pdf_dir)
                except requests.RequestException as exc:
                    logger.warning(
                        "Failed to download PDF for digest %s: %s", digest, exc
                    )
                    continue
                downloaded.append(digest)
                logger.info("Downloaded %d of %d PDFs", len(downloaded), args.num_pdfs)
            cdx_frames.append(cdx_rows[cdx_rows["digest"].isin(downloaded)])
            if len(downloaded) >= args.num_pdfs:
                break

    # The CDX sample and the manifest both cover every PDF in pdf_dir: rows for
    # new downloads, rows saved by earlier runs, and (rarely) a full-CDX lookup
    # for local PDFs that have no saved rows.
    pdf_digests = sorted(path.stem for path in args.pdf_dir.glob("*.pdf"))
    if pdf_digests:
        if cdx_path.exists():
            cdx_frames.append(
                pd.read_parquet(cdx_path, filters=[("digest", "in", pdf_digests)])
            )
        known_digests = {digest for frame in cdx_frames for digest in frame["digest"]}
        missing_digests = sorted(set(pdf_digests) - known_digests)
        if missing_digests:
            logger.info(
                "Looking up CDX rows for %d local PDFs (scans the full CDX)",
                len(missing_digests),
            )
            cdx_frames.append(_load_cdx_rows(connection, missing_digests))
        cdx = pd.concat(cdx_frames, ignore_index=True).drop_duplicates()
        cdx.to_parquet(cdx_path, index=False)

        manifest = (
            cdx.groupby("digest", as_index=False)["url"]
            .min()
            .assign(file=lambda frame: frame["digest"] + ".pdf")[
                ["file", "digest", "url"]
            ]
        )
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
