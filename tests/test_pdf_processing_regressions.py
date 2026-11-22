"""Output-regression tests for the PDF processing pipeline.

These complement the CodSpeed benchmarks in `tests/benchmarks/` (which track
*performance*): this module uses `pytest-regressions` to snapshot
deterministic pipeline *outputs* (extracted text, metadata, embedding shapes)
so unexpected changes in behavior are caught independently of timing noise.

If a change intentionally alters these outputs, regenerate the baselines with
`poetry run pytest tests/test_pdf_processing_regressions.py --force-regen` and
commit the updated files under
`tests/test_pdf_processing_regressions/`.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

import numpy as np

from govscape.config import DataModel
from govscape.processing import (
    PageImageEmbeddingStage,
    PDFExtractionStage,
    TextEmbeddingStage,
)

TEST_DATA_DIR = Path(__file__).resolve().parent / "test_data"
CDX_DIR = TEST_DATA_DIR / "cdx"
PDF_DIR = TEST_DATA_DIR / "pdfs"
CPU_COUNT = os.cpu_count() or 1


def _sample_pdf_files() -> list[str]:
    with open(CDX_DIR / "digests_manifest.csv", newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    return sorted(str(PDF_DIR / row["file"]) for row in manifest_rows)


@pytest.fixture(scope="module")
def processed_data_model(tmp_path_factory) -> tuple[DataModel, list[str]]:
    """Run extraction + (cheap, Dummy) embedding once for all regression checks."""
    pdf_files = _sample_pdf_files()
    data_dir = tmp_path_factory.mktemp("pdf_processing_regressions")
    data_model = DataModel(data_dir.as_posix())

    PDFExtractionStage(
        data_model=data_model, pdf_files=pdf_files, cpu_count=CPU_COUNT
    ).run()
    TextEmbeddingStage(data_model=data_model, model_type="Dummy").run()
    PageImageEmbeddingStage(
        data_model=data_model, model_type="Dummy", cpu_count=CPU_COUNT
    ).run()

    return data_model, pdf_files


def test_pdf_extraction_text_regression(processed_data_model, data_regression):
    """Extracted per-page text should not change unexpectedly."""
    data_model, pdf_files = processed_data_model

    extracted = {}
    for pdf_file in pdf_files:
        digest = Path(pdf_file).stem
        txt_dir = Path(data_model.txt_pdf_directory(digest))
        extracted[digest] = {
            page.name: page.read_text(encoding="utf-8")
            for page in sorted(txt_dir.glob("*.txt"))
        }

    data_regression.check(extracted)


def test_pdf_extraction_metadata_regression(processed_data_model, data_regression):
    """Extracted PDF metadata (title, page count, ...) should not change
    unexpectedly."""
    data_model, pdf_files = processed_data_model

    metadata = {}
    for pdf_file in pdf_files:
        digest = Path(pdf_file).stem
        with open(data_model.metadata_file_path(digest), encoding="utf-8") as f:
            metadata[digest] = json.load(f)

    data_regression.check(metadata)


def test_text_embedding_shape_regression(processed_data_model, data_regression):
    """The Dummy text embedding model returns random values, so only the shape
    and dtype of its output are regression-checked (not the values)."""
    data_model, pdf_files = processed_data_model

    shapes = {}
    for pdf_file in pdf_files:
        digest = Path(pdf_file).stem
        embed_files = sorted(
            Path(data_model.embedding_pdf_directory(digest)).glob("*.npy")
        )
        sample = np.load(embed_files[0]) if embed_files else None
        shapes[digest] = {
            "num_pages": len(embed_files),
            "dim": int(sample.shape[0]) if sample is not None else 0,
            "dtype": str(sample.dtype) if sample is not None else None,
        }

    data_regression.check(shapes)


def test_image_embedding_shape_regression(processed_data_model, data_regression):
    """The Dummy visual embedding model returns random values, so only the
    shape and dtype of its output are regression-checked (not the values)."""
    data_model, pdf_files = processed_data_model

    shapes = {}
    for pdf_file in pdf_files:
        digest = Path(pdf_file).stem
        embed_files = sorted(
            Path(data_model.embedding_img_pg_pdf_directory(digest)).glob("*.npy")
        )
        sample = np.load(embed_files[0]) if embed_files else None
        shapes[digest] = {
            "num_pages": len(embed_files),
            "dim": int(sample.shape[0]) if sample is not None else 0,
            "dtype": str(sample.dtype) if sample is not None else None,
        }

    data_regression.check(shapes)
