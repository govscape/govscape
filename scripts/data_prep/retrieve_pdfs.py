import logging
import multiprocessing
import os
import tempfile
from collections import defaultdict

import duckdb
import pyarrow.parquet as pq
import pypdfium2
import requests
from warcio.archiveiterator import ArchiveIterator

from govscape.data_loader import DataLoader, build_data_loader
from govscape.utils import base_argument_parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

_USER_AGENT = "govscape/0.1 (PDF Retrieval Script; kdeeds@cs.washington.edu)"

_worker_data_loader: DataLoader | None = None


def _init_worker(backend: str, bucket_name: str, local_base_dir: str) -> None:
    global _worker_data_loader
    _worker_data_loader = build_data_loader(backend, bucket_name, local_base_dir)


def _is_parseable_pdf(data: bytes) -> bool:
    try:
        doc = pypdfium2.PdfDocument(data)
        doc.close()
        return True
    except Exception:
        return False


def _iter_worker_args(parquet_path: str, output_dir: str, max_pdfs: int | None):
    pf = pq.ParquetFile(parquet_path)
    count = 0
    for batch in pf.iter_batches(columns=["digest", "filename", "offset", "length"]):
        for digest, filename, offset, length in zip(
            batch.column("digest").to_pylist(),
            batch.column("filename").to_pylist(),
            batch.column("offset").to_pylist(),
            batch.column("length").to_pylist(),
            strict=True,
        ):
            if max_pdfs is not None and count >= max_pdfs:
                return
            yield (digest, filename, int(offset), int(length), output_dir)
            count += 1


def _process_one_pdf(args: tuple) -> str:
    digest, filename, offset, length, output_directory = args
    assert _worker_data_loader is not None

    output_path = os.path.join(output_directory, digest + ".pdf")

    try:
        if _worker_data_loader.exists(output_path):
            return "skipped"
    except Exception:
        pass

    s3_url = f"https://eotarchive.s3.amazonaws.com/{filename}"
    byte_range = f"bytes={offset}-{offset + length - 1}"

    try:
        response = requests.get(
            s3_url,
            headers={"user-agent": _USER_AGENT, "Range": byte_range},
            stream=True,
        )
        for record in ArchiveIterator(response.raw):
            if record.rec_type != "response":
                continue
            is_pdf = (record.rec_headers.get("Content-Type") == "application/pdf") or (
                ".pdf" in record.rec_headers.get("WARC-Target-URI", "")
            )
            if not is_pdf:
                return "invalid_content"
            data = record.content_stream().read()
            if not _is_parseable_pdf(data):
                return "parse_failed"
            _worker_data_loader.upload_bytes(data, output_path)
            return "uploaded"
    except Exception as e:
        logging.warning("Error processing %s: %s", filename, e)
        return "error"

    return "invalid_content"


def main() -> None:
    parser = base_argument_parser(description="Retrieve PDFs from S3 & store them.")
    parser.add_argument(
        "--cdx_parquet",
        required=True,
        help="Remote key of CDX parquet file in the PDF archive",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Remote key prefix for output PDFs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count() * 4,
        help="Number of worker processes (default: cpu_count * 4)",
    )
    parser.add_argument(
        "--max_pdfs",
        type=int,
        default=None,
        help="Limit total PDFs processed (useful for testing)",
    )
    args = parser.parse_args()

    data_loader = build_data_loader(
        args.backend,
        args.pdf_bucket_name,
        args.local_base_dir,
        profile_name=args.pdf_profile,
    )
    with tempfile.TemporaryDirectory(prefix="retrieve_pdfs_") as tmp_dir:
        local_parquet = os.path.join(tmp_dir, "cdx.parquet")
        logging.info("Downloading %s", args.cdx_parquet)
        data_loader.download_file(args.cdx_parquet, local_parquet)

        deduped_parquet = os.path.join(tmp_dir, "cdx_deduped.parquet")
        logging.info("Deduplicating by digest")
        duckdb.execute(
            "COPY ("
            "  SELECT digest,"
            "    ANY_VALUE(filename) AS filename,"
            '    ANY_VALUE("offset") AS "offset",'
            '    ANY_VALUE("length") AS "length"'
            f"  FROM read_parquet('{local_parquet}')"
            "  GROUP BY digest"
            f") TO '{deduped_parquet}' (FORMAT PARQUET)"
        )

        total_rows = pq.read_metadata(deduped_parquet).num_rows
        num_total = (
            min(total_rows, args.max_pdfs) if args.max_pdfs is not None else total_rows
        )
        logging.info("CDX parquet has %d entries; processing %d", total_rows, num_total)

        num_workers = min(args.num_workers, num_total)
        logging.info("Processing %d PDFs across %d workers", num_total, num_workers)

        counts: dict[str, int] = defaultdict(int)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(args.backend, args.pdf_bucket_name, args.local_base_dir),
        ) as pool:
            for status in pool.imap_unordered(
                _process_one_pdf,
                _iter_worker_args(deduped_parquet, args.output_dir, args.max_pdfs),
            ):
                counts[status] += 1
                total = sum(counts.values())
                if total % 100 == 0:
                    logging.info("Progress: %d processed — %s", total, dict(counts))

    logging.info("Done. Final counts: %s", dict(counts))


if __name__ == "__main__":
    main()
