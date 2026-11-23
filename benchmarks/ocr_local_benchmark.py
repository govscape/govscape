"""Benchmark OCR engines on local PDFs, isolating engine compute.

Extracts page images from a local PDF directory once (cached), samples a fixed
page budget, then times each OCR engine's batched ``extract_text`` over the exact
same images. Reports pages/sec plus total characters extracted (a quality sanity
check so a "speedup" that silently drops text is visible).

It is used to prove that a change to an engine's batched interface is a real
speedup: run the same page set through several configs and compare.

Example:
    poetry run python -m benchmarks.ocr_local_benchmark \
        --engine easyocr --gpu --num-pdfs 25 --max-pages 40 \
        --easyocr-batch-sizes 1,8,16,32
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from govscape.config import DataModel
from govscape.processing.ocr import EasyOCRImpl, OcrMyPDFImpl, OLMOcrImpl
from govscape.processing.pdf_extraction_stage import PDFExtractionStage

ENGINE_CLASSES = {
    "easyocr": EasyOCRImpl,
    "olmocr": OLMOcrImpl,
    "ocrmypdf": OcrMyPDFImpl,
}


@dataclass
class RunResult:
    label: str
    pages: int
    median_seconds: float
    best_seconds: float
    pages_per_sec: float
    total_chars: int


def prepare_images(
    pdf_dir: Path, work_dir: Path, num_pdfs: int, max_pages: int, to_rgb: bool
) -> list[np.ndarray]:
    """Extract page images (cached) and return up to ``max_pages`` loaded images.

    Pages are sampled evenly across the extracted set so the budget spans many
    documents rather than just the first few.
    """
    import cv2

    data_model = DataModel(str(work_dir))
    pdf_files = sorted(str(p) for p in pdf_dir.glob("*.pdf"))[:num_pdfs]
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    if not os.path.isdir(data_model.image_directory) or not os.listdir(
        data_model.image_directory
    ):
        print(f"Extracting page images from {len(pdf_files)} PDFs into {work_dir} ...")
        stage = PDFExtractionStage(
            data_model=data_model,
            pdf_files=pdf_files,
            cpu_count=os.cpu_count() or 1,
        )
        stage.validate()
        stage.run()
    else:
        print(f"Reusing cached page images in {work_dir}")

    image_paths: list[str] = []
    for digest_dir in sorted(
        os.scandir(data_model.image_directory), key=lambda d: d.name
    ):
        if not digest_dir.is_dir():
            continue
        image_paths.extend(
            os.path.join(digest_dir.path, page_file)
            for page_file in sorted(os.listdir(digest_dir.path))
            if page_file.endswith(".jpeg")
        )

    if not image_paths:
        raise RuntimeError("No page images were produced by extraction.")

    # Evenly sample the page budget across all extracted pages.
    if len(image_paths) > max_pages:
        step = len(image_paths) / max_pages
        image_paths = [image_paths[int(i * step)] for i in range(max_pages)]

    images: list[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    print(f"Loaded {len(images)} page images for benchmarking.")
    return images


def time_engine(
    label: str,
    engine,
    images: list[np.ndarray],
    repeats: int,
) -> RunResult:
    """Warm up once (model load + lazy init excluded), then time ``repeats`` runs."""
    engine.validate()
    # Warmup: exercises model download/init and any first-call autotuning.
    engine.extract_text(images[: min(2, len(images))])

    per_run: list[float] = []
    total_chars = 0
    for _ in range(repeats):
        start = time.perf_counter()
        texts = engine.extract_text(images)
        per_run.append(time.perf_counter() - start)
        total_chars = sum(len(t) for t in texts)

    median = statistics.median(per_run)
    best = min(per_run)
    pages_per_sec = len(images) / median if median > 0 else float("inf")
    return RunResult(
        label=label,
        pages=len(images),
        median_seconds=median,
        best_seconds=best,
        pages_per_sec=pages_per_sec,
        total_chars=total_chars,
    )


def format_results(results: list[RunResult]) -> str:
    """Render the results table, with speedups relative to the first config."""
    header = (
        f"{'Config':<24} {'Pages':>6} {'Median s':>10} {'Best s':>9} "
        f"{'Pages/s':>9} {'Chars':>9} {'Speedup':>8}"
    )
    lines = [header, "-" * len(header)]
    baseline = results[0] if results else None
    for r in results:
        speedup = ""
        if baseline is not None and baseline.median_seconds > 0:
            speedup = f"{baseline.median_seconds / r.median_seconds:.2f}x"
        lines.append(
            f"{r.label:<24} {r.pages:>6} {r.median_seconds:>10.3f} "
            f"{r.best_seconds:>9.3f} {r.pages_per_sec:>9.2f} "
            f"{r.total_chars:>9} {speedup:>8}"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/test_data/TechnicalReport234PDFs"),
    )
    parser.add_argument("--engine", choices=list(ENGINE_CLASSES), default="easyocr")
    parser.add_argument("--num-pdfs", type=int, default=25)
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".ocr_bench_work"),
        help="Dir for extracted page images (cached across runs).",
    )
    parser.add_argument(
        "--easyocr-batch-sizes",
        default="1,16",
        help="Comma-separated recognition batch sizes to sweep for easyocr.",
    )
    parser.add_argument(
        "--ocrmypdf-workers",
        default="1,8",
        help="Comma-separated concurrency levels to sweep for ocrmypdf.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    images = prepare_images(
        args.pdf_dir, args.work_dir, args.num_pdfs, args.max_pages, to_rgb=True
    )

    if args.engine == "easyocr":
        configs = [
            (f"easyocr bs={bs}", {"gpu": args.gpu, "batch_size": int(bs)})
            for bs in args.easyocr_batch_sizes.split(",")
        ]
    elif args.engine == "ocrmypdf":
        configs = [
            (f"ocrmypdf workers={w}", {"max_workers": int(w)})
            for w in args.ocrmypdf_workers.split(",")
        ]
    else:
        configs = [(args.engine, {})]

    print(f"Benchmarking engine={args.engine} gpu={args.gpu} over {len(images)} pages")
    results: list[RunResult] = []
    for label, kwargs in configs:
        print(f"  running: {label} ...")
        engine = ENGINE_CLASSES[args.engine](**kwargs)
        results.append(time_engine(label, engine, images, args.repeats))

    print()
    print(format_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
