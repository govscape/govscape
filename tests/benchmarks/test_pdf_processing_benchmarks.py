"""CodSpeed benchmarks for the PDF processing pipeline.

Run locally with `poetry run pytest tests/benchmarks --codspeed`.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from govscape.config import DataModel
from govscape.pdf_processing_pipeline import PDFProcessingPipeline
from govscape.processing import (
    PageImageEmbeddingStage,
    PDFExtractionStage,
    TextEmbeddingStage,
)
from govscape.processing.pdf_extraction_stage import _convert_single_pdf

from .conftest import _timed_stage

CPU_COUNT = os.cpu_count() or 1


def _fresh_data_model(base_dir: Path) -> DataModel:
    return DataModel((base_dir / uuid.uuid4().hex).as_posix())


def test_single_pdf_conversion_benchmark(benchmark, sample_pdf_files, tmp_path: Path):
    """Benchmark converting one PDF to text/images/metadata, in-process.

    Calls the same per-PDF conversion logic `PDFExtractionStage` dispatches to
    its worker pool, but in-process (no `multiprocessing` spawn), isolating
    the actual parsing cost from process-startup/import overhead.
    """
    pdf_file = sample_pdf_files[0]

    def run():
        data_model = _fresh_data_model(tmp_path)
        ok = _convert_single_pdf(data_model, pdf_file)
        shutil.rmtree(data_model.data_dir, ignore_errors=True)
        return ok

    assert benchmark(lambda: _timed_stage("single PDF conversion", run)) is True


def test_pdf_extraction_benchmark(benchmark, sample_pdf_files, tmp_path: Path):
    """Benchmark converting the full PDF sample to per-page text, images, and
    metadata."""

    def run():
        data_model = _fresh_data_model(tmp_path)
        stage = PDFExtractionStage(
            data_model=data_model, pdf_files=sample_pdf_files, cpu_count=CPU_COUNT
        )
        parsed_count = stage.run()
        shutil.rmtree(data_model.data_dir, ignore_errors=True)
        return parsed_count

    name = f"PDF extraction ({len(sample_pdf_files)} files)"
    parsed_count = benchmark.pedantic(
        lambda: _timed_stage(name, run), rounds=1, warmup_rounds=0, iterations=1
    )
    assert parsed_count == len(sample_pdf_files)


def test_end_to_end_pipeline_benchmark(benchmark, sample_pdf_files, tmp_path: Path):
    """Benchmark the full PDF -> text/image -> embeddings pipeline (no OCR),
    using the real ST/CLIP models (see module docstring)."""

    def run():
        data_model = _fresh_data_model(tmp_path)
        pipeline = PDFProcessingPipeline(
            data_model.data_dir, text_model_type="ST", visual_model_type="CLIP"
        )
        # pipeline.process_pdfs() already logs a line per internal stage
        # (extraction, text embedding, image embedding) as it completes each.
        timings = pipeline.process_pdfs(
            sample_pdf_files,
            do_text_embedding=True,
            do_img_embedding=True,
            do_metadata_collection=True,
            do_ocr=False,
        )
        shutil.rmtree(data_model.data_dir, ignore_errors=True)
        return timings

    timings = benchmark.pedantic(
        lambda: _timed_stage("end-to-end pipeline", run),
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
    assert len(timings) == 4


def test_text_embedding_benchmark(benchmark, sample_pdf_files, tmp_path_factory):
    """Benchmark embedding the extracted page text with the real ST model.

    The `TextEmbeddingStage` (and its model) is built once, outside the
    benchmarked callable, so the measured call isolates encoding throughput
    from the one-time model download/load cost.
    """
    data_dir = tmp_path_factory.mktemp("text_embedding_benchmark")
    data_model = DataModel(data_dir.as_posix())
    _timed_stage(
        "extraction setup (not benchmarked)",
        PDFExtractionStage(
            data_model=data_model, pdf_files=sample_pdf_files, cpu_count=CPU_COUNT
        ).run,
    )
    stage = _timed_stage(
        "loading ST text embedding model (not benchmarked; may download weights)",
        lambda: TextEmbeddingStage(data_model=data_model, model_type="ST"),
    )

    benchmark.pedantic(
        lambda: _timed_stage("text embedding", stage.run),
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )


def test_image_embedding_benchmark(benchmark, sample_pdf_files, tmp_path_factory):
    """Benchmark embedding the extracted page images with the real CLIP model.

    The `PageImageEmbeddingStage` (and its model) is built once, outside the
    benchmarked callable, so the measured call isolates encoding throughput
    from the one-time model download/load cost.
    """
    data_dir = tmp_path_factory.mktemp("image_embedding_benchmark")
    data_model = DataModel(data_dir.as_posix())
    _timed_stage(
        "extraction setup (not benchmarked)",
        PDFExtractionStage(
            data_model=data_model, pdf_files=sample_pdf_files, cpu_count=CPU_COUNT
        ).run,
    )
    stage = _timed_stage(
        "loading CLIP visual embedding model (not benchmarked; may download weights)",
        lambda: PageImageEmbeddingStage(
            data_model=data_model, model_type="CLIP", cpu_count=CPU_COUNT
        ),
    )

    benchmark.pedantic(
        lambda: _timed_stage("image embedding", stage.run),
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
