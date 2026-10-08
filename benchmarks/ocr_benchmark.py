"""Benchmark OCR processing performance using real PDFs from S3.

Downloads PDFs from the source.coop PDF archive, extracts page images via
PDFExtractionStage, then times each OCR engine.

Example:
    poetry run python -m govscape.benchmarks.ocr_benchmark \
        --num-pdfs 10 --engines easyocr paddleocr
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from botocore.config import Config

from govscape.config import (
    PDF_ARCHIVE_BUCKET,
    PDF_ARCHIVE_PDF_DIR,
    SOURCE_COOP_ENDPOINT,
    DataModel,
)
from govscape.data_loader import S3DataLoader
from govscape.processing.ocr_processing_stage import OCRProcessingStage
from govscape.processing.pdf_extraction_stage import PDFExtractionStage

DEFAULT_BUCKET = PDF_ARCHIVE_BUCKET
DEFAULT_PREFIX = PDF_ARCHIVE_PDF_DIR
DEFAULT_NUM_PDFS = 10
DEFAULT_ENGINES = ["easyocr", "paddleocr", "olmocr", "ocrmypdf"]

ENGINE_DEFAULT_KWARGS: dict[str, dict] = {
    "easyocr": {"languages": ["en"], "gpu": False},
    "paddleocr": {"language": "en", "use_gpu": False},
    "olmocr": {"model_name": "default"},
    "ocrmypdf": {"language": "eng", "output_type": "txt"},
}


def _apply_gpu(kwargs: dict[str, dict], gpu: bool) -> dict[str, dict]:
    updated = {k: dict(v) for k, v in kwargs.items()}
    updated["easyocr"]["gpu"] = gpu
    updated["paddleocr"]["use_gpu"] = gpu
    return updated


@dataclass
class BenchmarkResult:
    engine: str
    total_pages: int
    seconds: float
    pages_per_sec: float
    error: str | None = None


def download_pdfs(data_root: str, bucket: str, prefix: str, num_pdfs: int) -> list[str]:
    """Download up to num_pdfs PDFs from S3 and return their local paths."""
    loader = S3DataLoader(
        bucket_name=bucket,
        config=Config(max_pool_connections=60),
        endpoint_url=SOURCE_COOP_ENDPOINT,
    )
    pdf_dir = os.path.join(data_root, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)

    local_paths: list[str] = []
    continuation_token = None
    while len(local_paths) < num_pdfs:
        result = loader.list_objects(
            prefix=prefix,
            max_keys=num_pdfs - len(local_paths),
            continuation_token=continuation_token,
        )
        for key in result.keys:
            if key.endswith(".pdf"):
                local_path = os.path.join(pdf_dir, os.path.basename(key))
                loader.download_file(key, local_path)
                local_paths.append(local_path)
                if len(local_paths) >= num_pdfs:
                    break
        if not result.is_truncated:
            break
        continuation_token = result.continuation_token

    return local_paths


def extract_images(data_model: DataModel, pdf_files: list[str]) -> int:
    """Extract page images from PDFs and return total image count."""
    stage = PDFExtractionStage(
        data_model=data_model,
        pdf_files=pdf_files,
        cpu_count=os.cpu_count() or 1,
    )
    stage.validate()
    stage.run()

    total = 0
    if os.path.isdir(data_model.image_directory):
        for digest_dir in os.scandir(data_model.image_directory):
            if digest_dir.is_dir():
                total += sum(
                    1 for f in os.listdir(digest_dir.path) if f.endswith(".jpeg")
                )
    return total


def select_engines(requested: Sequence[str]) -> list[str]:
    if not requested:
        return DEFAULT_ENGINES
    selected = []
    for engine in requested:
        normalized = engine.lower()
        if normalized not in DEFAULT_ENGINES:
            raise ValueError(
                f"Unknown engine '{engine}'. "
                f"Supported engines: {', '.join(DEFAULT_ENGINES)}"
            )
        selected.append(normalized)
    return selected


def benchmark_engine(
    engine: str,
    data_model: DataModel,
    total_pages: int,
    engine_kwargs: dict[str, dict] | None = None,
) -> BenchmarkResult:
    engine_kwargs = (engine_kwargs or ENGINE_DEFAULT_KWARGS).get(engine, {})
    try:
        stage = OCRProcessingStage(
            data_model=data_model, ocr_type=engine, **engine_kwargs
        )
        stage.validate()

        if os.path.isdir(data_model.txt_directory):
            shutil.rmtree(data_model.txt_directory)

        start = time.perf_counter()
        stage.run()
        runtime = time.perf_counter() - start
        pages_per_sec = total_pages / runtime if runtime > 0 else float("inf")
        return BenchmarkResult(
            engine=engine,
            total_pages=total_pages,
            seconds=runtime,
            pages_per_sec=pages_per_sec,
        )
    except Exception as error:
        return BenchmarkResult(
            engine=engine,
            total_pages=total_pages,
            seconds=0.0,
            pages_per_sec=0.0,
            error=str(error),
        )


def format_results(results: list[BenchmarkResult]) -> str:
    header = (
        f"{'Engine':<12} {'Pages':>6} {'Seconds':>10} {'Pages/s':>10} {'Status':>8}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        status = "OK" if result.error is None else "FAILED"
        lines.append(
            f"{result.engine:<12} {result.total_pages:>6} {result.seconds:>10.4f} "
            f"{result.pages_per_sec:>10.2f} {status:>8}"
        )
        if result.error is not None:
            lines.append(f"  Error: {result.error}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OCR processing performance."
    )
    parser.add_argument(
        "--num-pdfs",
        type=int,
        default=DEFAULT_NUM_PDFS,
        help="Number of PDFs to download from S3.",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="S3 bucket name.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="S3 key prefix for PDFs.",
    )
    parser.add_argument(
        "--engines",
        nargs="*",
        default=None,
        help="Subset of OCR engines to benchmark.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./.ocr_benchmark_data"),
        help="Local directory for downloaded PDFs and extracted images.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep benchmark data after the run.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration for EasyOCR and PaddleOCR.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if os.path.isdir(args.data_root) and not args.keep_data:
        shutil.rmtree(args.data_root, ignore_errors=True)
    os.makedirs(args.data_root, exist_ok=True)

    data_model = DataModel(str(args.data_root))

    print(f"Downloading {args.num_pdfs} PDFs from s3://{args.bucket}/{args.prefix} ...")
    pdf_files = download_pdfs(
        str(args.data_root), args.bucket, args.prefix, args.num_pdfs
    )
    print(f"Downloaded {len(pdf_files)} PDFs. Extracting page images ...")
    total_pages = extract_images(data_model, pdf_files)
    print(f"Extracted {total_pages} page images.")

    engines = select_engines(args.engines or [])
    engine_kwargs = _apply_gpu(ENGINE_DEFAULT_KWARGS, args.gpu)
    results: list[BenchmarkResult] = []

    for engine in engines:
        print(f"Running OCR benchmark for engine: {engine}")
        results.append(benchmark_engine(engine, data_model, total_pages, engine_kwargs))

    print(format_results(results))

    if not args.keep_data:
        shutil.rmtree(args.data_root, ignore_errors=True)

    return 0 if all(result.error is None for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
