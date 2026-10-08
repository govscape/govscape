"""Benchmark visual embedding model throughput on local PDF page images.

Extracts page images from a local PDF directory once (cached), then for each
model reports:

* end-to-end images/sec through ``encode_images`` (CPU preprocessing + GPU), the
  path used by the embedding pipeline,
* GPU-only images/sec on already-preprocessed pixels,
* median latency of ``encode_text`` for short search queries (the serving path),
* peak GPU memory.

Models are given as ``clip`` or ``siglip:<huggingface model name>``.

Example:
    poetry run python -m benchmarks.visual_embedding_benchmark \
        --models clip siglip:google/siglip-base-patch16-224 \
        --max-pages 1024
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from govscape.config import DataModel
from govscape.processing.pdf_extraction_stage import PDFExtractionStage
from govscape.visual_embedding_models import (
    CLIP_VisualEmbeddingModel,
    SigLIP_VisualEmbeddingModel,
)

QUERIES = [
    "map of national parks",
    "budget table for fiscal year 2020",
    "photograph of a bridge",
    "organizational chart",
    "wildfire risk assessment",
    "bar chart of unemployment rates",
    "hurricane evacuation routes",
    "signature page of a contract",
]


@dataclass
class ModelResult:
    label: str
    dim: int
    load_seconds: float
    e2e_images_per_sec: float
    gpu_images_per_sec: float
    query_ms: float
    peak_gpu_mib: float


def build_model(spec: str):
    if spec == "clip":
        return CLIP_VisualEmbeddingModel()
    if spec.startswith("siglip:"):
        return SigLIP_VisualEmbeddingModel(spec.split(":", 1)[1])
    raise ValueError(f"Unknown model spec: {spec}")


def prepare_image_paths(
    pdf_dir: Path, work_dir: Path, num_pdfs: int, max_pages: int
) -> list[str]:
    data_model = DataModel(str(work_dir))
    if not os.path.isdir(data_model.image_directory) or not os.listdir(
        data_model.image_directory
    ):
        pdf_files = sorted(str(p) for p in pdf_dir.glob("*.pdf"))[:num_pdfs]
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
        print(f"Extracting page images from {len(pdf_files)} PDFs into {work_dir} ...")
        stage = PDFExtractionStage(
            data_model=data_model,
            pdf_files=pdf_files,
            cpu_count=os.cpu_count() or 1,
        )
        stage.validate()
        stage.run()

    paths = sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(data_model.image_directory)
        for name in files
        if name.endswith(".jpeg")
    )
    if not paths:
        raise RuntimeError("No page images were produced by extraction.")
    # Repeat pages if needed so every run sees a full budget of images.
    while len(paths) < max_pages:
        paths = paths + paths
    return paths[:max_pages]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_gpu_only(
    model, image_paths: list[str], batch_size: int, repeats: int
) -> float:
    pixels = model._load_and_process_image(image_paths, model.processor)
    timings = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        for i in range(0, pixels.size(0), batch_size):
            batch = pixels[i : i + batch_size].to(model.device)
            with torch.no_grad():
                out = model.model.get_image_features(pixel_values=batch)
                out = out / out.norm(dim=-1, keepdim=True)
            out.cpu()
        _sync()
        timings.append(time.perf_counter() - start)
    return pixels.size(0) / statistics.median(timings)


def benchmark_model(
    spec: str, image_paths: list[str], repeats: int, batch_size: int
) -> ModelResult:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    model = build_model(spec)
    load_seconds = time.perf_counter() - start

    # Warmup: first-call CUDA/kernel initialisation is excluded from timings.
    model.encode_images(image_paths[: min(8, len(image_paths))])
    model.encode_text(QUERIES[0])

    e2e = []
    for _ in range(repeats):
        _sync()
        start = time.perf_counter()
        model.encode_images(image_paths)
        _sync()
        e2e.append(time.perf_counter() - start)

    gpu_ips = time_gpu_only(model, image_paths, batch_size, repeats)

    query_times = []
    for _ in range(repeats):
        for q in QUERIES:
            _sync()
            start = time.perf_counter()
            model.encode_text(q)
            _sync()
            query_times.append(time.perf_counter() - start)

    peak = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    result = ModelResult(
        label=spec,
        dim=model.d,
        load_seconds=load_seconds,
        e2e_images_per_sec=len(image_paths) / statistics.median(e2e),
        gpu_images_per_sec=gpu_ips,
        query_ms=statistics.median(query_times) * 1000,
        peak_gpu_mib=peak,
    )
    del model
    return result


def format_results(results: list[ModelResult]) -> str:
    header = (
        f"{'Model':<42} {'Dim':>5} {'Load s':>7} {'E2E img/s':>10} "
        f"{'GPU img/s':>10} {'Query ms':>9} {'Peak MiB':>9} {'E2E vs 1st':>10}"
    )
    lines = [header, "-" * len(header)]
    base = results[0].e2e_images_per_sec if results else 0
    lines.extend(
        f"{r.label:<42} {r.dim:>5} {r.load_seconds:>7.1f} "
        f"{r.e2e_images_per_sec:>10.1f} {r.gpu_images_per_sec:>10.1f} "
        f"{r.query_ms:>9.2f} {r.peak_gpu_mib:>9.0f} "
        f"{r.e2e_images_per_sec / base:>9.2f}x"
        for r in results
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=Path("tests/test_data/pdfs"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clip", f"siglip:{SigLIP_VisualEmbeddingModel.DEFAULT_MODEL_NAME}"],
    )
    parser.add_argument("--num-pdfs", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".visual_bench_work"),
        help="Dir for extracted page images (cached across runs).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    image_paths = prepare_image_paths(
        args.pdf_dir, args.work_dir, args.num_pdfs, args.max_pages
    )
    print(f"Benchmarking on {len(image_paths)} page images")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    results = []
    for spec in args.models:
        print(f"== {spec}")
        results.append(
            benchmark_model(spec, image_paths, args.repeats, args.batch_size)
        )
    print()
    print(format_results(results))


if __name__ == "__main__":
    main()
