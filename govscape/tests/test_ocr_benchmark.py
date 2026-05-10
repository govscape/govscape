# AI modified: 2026-05-10 c0b26991
"""Smoke tests for the OCR benchmark script."""

import importlib.util
import sys
from pathlib import Path

import pytest


def load_benchmark_module():
    benchmark_path = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "ocr_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("ocr_benchmark", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ocr_benchmark"] = module
    spec.loader.exec_module(module)
    return module


def load_local_config_module():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "govscape" / "config.py"
    spec = importlib.util.spec_from_file_location("local_config", config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_config"] = module
    spec.loader.exec_module(module)
    return module


def test_select_engines_default_and_invalid():
    benchmark = load_benchmark_module()

    assert benchmark.select_engines([]) == benchmark.DEFAULT_ENGINES
    assert benchmark.select_engines(["easyocr", "paddleocr"]) == [
        "easyocr",
        "paddleocr",
    ]

    with pytest.raises(ValueError, match="Unknown engine"):
        benchmark.select_engines(
            [
                "unsupported_engine",
            ]
        )


def test_generate_image_dataset_creates_images(tmp_path):
    benchmark = load_benchmark_module()
    config = load_local_config_module()
    DataModel = config.DataModel

    data_root = tmp_path / "ocr_benchmark_data"
    model = DataModel(str(data_root))

    total_pages = benchmark.generate_image_dataset(
        data_model=model,
        documents=2,
        pages_per_document=2,
        width=256,
        height=256,
        seed=123,
    )

    assert total_pages == 4
    assert (data_root / "img" / "doc_000000" / "doc_000000_0.jpeg").exists()
    assert (data_root / "img" / "doc_000001" / "doc_000001_1.jpeg").exists()


def test_main_runs_with_mocked_stage(tmp_path, monkeypatch):
    benchmark = load_benchmark_module()

    data_root = tmp_path / "ocr_benchmark_data"

    class FakeStage:
        def __init__(self, data_model, ocr_type, **kwargs):
            self.data_model = data_model
            self.ocr_type = ocr_type

        def validate(self):
            return None

        def run(self):
            return None

    monkeypatch.setattr(benchmark, "OCRProcessingStage", FakeStage)

    exit_code = benchmark.main(
        [
            "--documents",
            "1",
            "--pages-per-document",
            "1",
            "--width",
            "128",
            "--height",
            "128",
            "--data-root",
            str(data_root),
            "--keep-data",
        ]
    )

    assert exit_code == 0
    assert (data_root / "img" / "doc_000000" / "doc_000000_0.jpeg").exists()
