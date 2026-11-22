"""Bulk-download PDFs directly via HTTP from a source.coop repository.

Unlike ``download_sample_pdfs.py`` (which reads the bucket via the S3 API and
therefore needs AWS credentials), this script talks to the public
``data.source.coop`` HTTP endpoint directly, so it works with no credentials
at all. It is intended for exercising/testing the bulk-download pipeline
against real data, e.g.:

    poetry run python scripts/data_prep/download_source_coop_pdfs.py \
        --account govscape \
        --repository eota-pdf-archive \
        --local_base_dir data/s3_mock \
        --num_pdfs 1000

PDFs for this repository are addressed by digest, e.g.:
    https://data.source.coop/govscape/eota-pdf-archive/pdfs/22223HMLFQ7P5VQVLKZR3PZAI7YMGOXC.pdf

The CDX parquet file (``cdx/complete_cdx.parquet``) is many GB, so rather than
downloading it outright we query it in place over HTTP range requests via
DuckDB's httpfs extension, pulling only the handful of candidate rows we need.
"""

import argparse
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import requests

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)

_USER_AGENT = "govscape/0.1 (Bulk Download Test; kdeeds@cs.washington.edu)"


def repository_url(account: str, repository: str) -> str:
    return f"https://data.source.coop/{account}/{repository}"


def pdf_download_url(account: str, repository: str, digest: str) -> str:
    return f"{repository_url(account, repository)}/pdfs/{digest}.pdf"


def fetch_candidate_digests(account: str, repository: str, limit: int) -> list[dict]:
    """Query the repository's CDX parquet over HTTP range requests (via
    DuckDB's httpfs extension) for a sample of candidate digests with their
    crawl metadata, without downloading the full (multi-GB) file."""
    cdx_url = f"{repository_url(account, repository)}/cdx/complete_cdx.parquet"
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    rows = con.execute(
        f"""
        SELECT digest, filename, "offset", length, url, crawl_date
        FROM read_parquet('{cdx_url}')
        LIMIT ?
        """,
        [limit],
    ).fetchall()
    con.close()

    seen: set[str] = set()
    candidates = []
    for digest, filename, offset, length, url, crawl_date in rows:
        if digest in seen:
            continue
        seen.add(digest)
        candidates.append(
            {
                "digest": digest,
                "filename": filename,
                "offset": offset,
                "length": length,
                "url": url,
                "crawl_date": crawl_date,
            }
        )
    return candidates


def _download_one(
    session: requests.Session,
    account: str,
    repository: str,
    digest: str,
    dest_path: str,
) -> str:
    if os.path.exists(dest_path):
        return "skipped"
    pdf_url = pdf_download_url(account, repository, digest)
    try:
        response = session.get(pdf_url, headers={"user-agent": _USER_AGENT}, timeout=60)
    except requests.RequestException as exc:
        logging.warning("Error downloading %s: %s", digest, exc)
        return "error"
    if response.status_code != 200:
        logging.warning(
            "PDF for digest %s not found (status %d), skipping",
            digest,
            response.status_code,
        )
        return "not_found"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(response.content)
    return "downloaded"


def write_cdx_metadata_parquet(candidates: list[dict], output_path: str) -> None:
    """Write a CDX metadata parquet file (matching the schema documented in
    DATA_MODEL.md) for the given candidates, for use by
    generate_index_metadata.py."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE cdx ("
        'digest VARCHAR, filename VARCHAR, "offset" BIGINT, '
        "length BIGINT, url VARCHAR, crawl_date VARCHAR)"
    )
    con.executemany(
        "INSERT INTO cdx VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                c["digest"],
                c["filename"],
                c["offset"],
                c["length"],
                c["url"],
                c["crawl_date"],
            )
            for c in candidates
        ],
    )
    con.execute(f"COPY cdx TO '{output_path}' (FORMAT PARQUET)")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-download PDFs via HTTP from a source.coop repository."
    )
    parser.add_argument(
        "--account", default="govscape", help="source.coop account/org name"
    )
    parser.add_argument(
        "--repository", default="eota-pdf-archive", help="source.coop repository name"
    )
    parser.add_argument(
        "--local_base_dir",
        default="data/s3_mock",
        help="Local directory mirroring the repository structure",
    )
    parser.add_argument(
        "--num_pdfs",
        type=int,
        default=1000,
        help="Number of PDFs to download",
    )
    parser.add_argument(
        "--candidate_multiplier",
        type=int,
        default=2,
        help=(
            "Fetch this many candidate CDX rows per requested PDF, so there "
            "are replacements available for any digests that fail to download."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Number of concurrent downloads"
    )
    args = parser.parse_args()

    logging.info("Repository: %s", repository_url(args.account, args.repository))

    candidate_limit = max(args.num_pdfs * args.candidate_multiplier, args.num_pdfs)
    logging.info("Querying CDX parquet for up to %d candidate digests", candidate_limit)
    candidates = fetch_candidate_digests(args.account, args.repository, candidate_limit)
    logging.info("Found %d distinct candidate digests", len(candidates))

    pdf_dir = os.path.join(args.local_base_dir, "archive", "PDFs")
    os.makedirs(pdf_dir, exist_ok=True)

    counts: dict[str, int] = defaultdict(int)
    downloaded_candidates: list[dict] = []
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=args.workers, pool_maxsize=args.workers
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for candidate in candidates:
            dest_path = os.path.join(pdf_dir, f"{candidate['digest']}.pdf")
            future = executor.submit(
                _download_one,
                session,
                args.account,
                args.repository,
                candidate["digest"],
                dest_path,
            )
            futures[future] = candidate

        for future in as_completed(futures):
            candidate = futures[future]
            status = future.result()
            counts[status] += 1
            if status in ("downloaded", "skipped"):
                downloaded_candidates.append(candidate)
            total = sum(counts.values())
            if total % 100 == 0:
                logging.info("Progress: %d processed — %s", total, dict(counts))
            if len(downloaded_candidates) >= args.num_pdfs:
                for pending in futures:
                    pending.cancel()
                break

    downloaded_candidates = downloaded_candidates[: args.num_pdfs]
    logging.info(
        "PDF download complete: %d files in %s — %s",
        len(downloaded_candidates),
        pdf_dir,
        dict(counts),
    )

    cdx_output_path = os.path.join(
        args.local_base_dir, "archive", "CDX", "pdf_metadata.parquet"
    )
    write_cdx_metadata_parquet(downloaded_candidates, cdx_output_path)
    logging.info("Wrote CDX metadata for downloaded PDFs to %s", cdx_output_path)


if __name__ == "__main__":
    main()
