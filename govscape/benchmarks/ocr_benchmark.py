# AI modified: 2026-05-10 c0b26991
"""Benchmark OCR processing performance for synthetic PDF page images.

This benchmark generates a temporary image dataset using the repository's
DataModel layout, then runs the OCR processing stage for each selected OCR
backend.

Example:
    poetry run python -m govscape.benchmarks.ocr_benchmark \
        --documents 5 --pages-per-document 3 --engines easyocr paddleocr
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import shutil
import sys
import time
import types
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np

    import cv2
    from PIL import Image, ImageDraw, ImageFont

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


def _load_local_govscape_modules() -> tuple[type[Any], type[Any]]:
    repo_root = Path(__file__).resolve().parents[1]
    govscape_root = repo_root / "govscape"

    if "govscape" not in sys.modules:
        govscape_pkg = types.ModuleType("govscape")
        govscape_pkg.__path__ = [str(govscape_root)]
        sys.modules["govscape"] = govscape_pkg

    if "govscape.processing" not in sys.modules:
        processing_pkg = types.ModuleType("govscape.processing")
        processing_pkg.__path__ = [str(govscape_root / "processing")]
        sys.modules["govscape.processing"] = processing_pkg

    if "govscape.processing.ocr" not in sys.modules:
        ocr_pkg = types.ModuleType("govscape.processing.ocr")
        ocr_pkg.__path__ = [str(govscape_root / "processing" / "ocr")]
        sys.modules["govscape.processing.ocr"] = ocr_pkg

    def _load_module(module_name: str, module_path: Path):
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    config_module = _load_module("govscape.config", govscape_root / "config.py")
    ocr_module = _load_module(
        "govscape.processing.ocr_processing_stage",
        govscape_root / "processing" / "ocr_processing_stage.py",
    )
    return cast(type[Any], config_module.DataModel), cast(
        type[Any], ocr_module.OCRProcessingStage
    )


DataModel, OCRProcessingStage = _load_local_govscape_modules()

DEFAULT_IMAGE_WIDTH = 768
DEFAULT_IMAGE_HEIGHT = 1024
DEFAULT_DOCUMENTS = 5
DEFAULT_PAGES_PER_DOCUMENT = 2
DEFAULT_ENGINES = ["easyocr", "paddleocr", "olmocr", "ocrmypdf"]

ENGINE_DEFAULT_KWARGS: dict[str, dict[str, Any]] = {
    "easyocr": {"languages": ["en"], "gpu": False},
    "paddleocr": {"language": "en", "use_gpu": False},
    "olmocr": {"model_name": "default"},
    "ocrmypdf": {"language": "eng", "output_type": "txt"},
}


@dataclass
class BenchmarkResult:
    engine: str
    documents: int
    pages_per_document: int
    total_pages: int
    width: int
    height: int
    seconds: float
    pages_per_sec: float
    error: str | None = None


def _build_text_image(text: str, width: int, height: int) -> Any:
    if Image is not None and ImageDraw is not None and ImageFont is not None:
        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.load_default()
        except OSError:
            font = None

        margin = 20
        line_spacing = 8
        y: int = margin

        for line in text.splitlines():
            if font is not None:
                draw.text((margin, y), line, fill=(0, 0, 0), font=font)
                bbox = draw.textbbox((margin, y), line, font=font)
                line_height = int(bbox[3] - bbox[1])
            else:
                draw.text((margin, y), line, fill=(0, 0, 0))
                line_height = 12
            y += line_height + line_spacing
            if y > height - margin:
                break

        return image

    if cv2 is not None:
        if np is None:
            raise ImportError(
                "NumPy is required to generate images with OpenCV. "
                "Install it with: pip install numpy"
            )

        image = np.full((height, width, 3), 255, dtype=np.uint8)
        margin = 20
        line_spacing = 24
        y = margin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        for line in text.splitlines():
            cv2.putText(
                image,
                line,
                (margin, y),
                font,
                scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )
            y += int(18 * scale) + line_spacing
            if y > height - margin:
                break

        return image

    raise ImportError(
        "Pillow or OpenCV is required to generate synthetic OCR images. "
        "Install pillow with `pip install pillow` or "
        "opencv-python with `pip install opencv-python`."
    )


def _make_page_text(digest: str, page_no: int, seed: int) -> str:
    rng = random.Random(seed + page_no)
    words = [f"word{rng.randint(10, 999)}" for _ in range(10)]
    lines = [f"Document: {digest}", f"Page: {page_no + 1}", ""]
    lines.extend(" ".join(words[i : i + 5]) for i in range(0, len(words), 5))
    return "\n".join(lines)


def generate_image_dataset(
    data_model: Any,
    documents: int,
    pages_per_document: int,
    width: int,
    height: int,
    seed: int,
) -> int:
    if documents <= 0 or pages_per_document <= 0:
        raise ValueError("documents and pages_per_document must both be positive")

    if os.path.isdir(data_model.image_directory):
        shutil.rmtree(data_model.image_directory)
    os.makedirs(data_model.image_directory, exist_ok=True)

    total_pages = 0
    rng = random.Random(seed)

    for doc_index in range(documents):
        digest = f"doc_{doc_index:06d}"
        digest_dir = data_model.img_pdf_directory(digest)
        os.makedirs(digest_dir, exist_ok=True)

        for page_no in range(pages_per_document):
            page_text = _make_page_text(digest, page_no, seed + rng.randint(0, 9999))
            image = _build_text_image(page_text, width, height)
            image_path = data_model.img_page_path(digest, page_no)
            if cv2 is not None and isinstance(image, np.ndarray):
                cv2.imwrite(image_path, image)
            else:
                image.save(image_path, format="JPEG", quality=90)
            total_pages += 1

    return total_pages


def select_engines(requested: Sequence[str]) -> list[str]:
    if not requested:
        return DEFAULT_ENGINES

    selected = []
    for engine in requested:
        normalized = engine.lower()
        if normalized not in DEFAULT_ENGINES:
            supported = ", ".join(DEFAULT_ENGINES)
            raise ValueError(
                f"Unknown engine '{engine}'. Supported engines: {supported}"
            )
        selected.append(normalized)
    return selected


def benchmark_engine(
    engine: str,
    data_model: Any,
    documents: int,
    pages_per_document: int,
    width: int,
    height: int,
) -> BenchmarkResult:
    total_pages = documents * pages_per_document
    engine_kwargs = ENGINE_DEFAULT_KWARGS.get(engine, {})

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
            documents=documents,
            pages_per_document=pages_per_document,
            total_pages=total_pages,
            width=width,
            height=height,
            seconds=runtime,
            pages_per_sec=pages_per_sec,
        )
    except Exception as error:  # pylint: disable=broad-except
        return BenchmarkResult(
            engine=engine,
            documents=documents,
            pages_per_document=pages_per_document,
            total_pages=total_pages,
            width=0,
            height=0,
            seconds=0.0,
            pages_per_sec=0.0,
            error=str(error),
        )


def format_results(results: list[BenchmarkResult], width: int, height: int) -> str:
    header = (
        f"{'Engine':<12} {'Docs':>5} {'Pages':>5} {'Width':>6} {'Height':>6} "
        f"{'Seconds':>10} {'Pages/s':>10} {'Status':>12}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        status = "OK" if result.error is None else "FAILED"
        lines.append(
            f"{result.engine:<12} {result.documents:>5} {result.total_pages:>5} "
            f"{width:>6} {height:>6} {result.seconds:>10.4f} "
            f"{result.pages_per_sec:>10.2f} {status:>12}"
        )
        if result.error is not None:
            lines.append(f"Error: {result.error}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OCR processing performance."
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=DEFAULT_DOCUMENTS,
        help="Number of synthetic documents to generate.",
    )
    parser.add_argument(
        "--pages-per-document",
        type=int,
        default=DEFAULT_PAGES_PER_DOCUMENT,
        help="Number of pages to generate per document.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Synthetic page image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Synthetic page image height in pixels.",
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
        help="Temporary data root for synthetic OCR dataset.",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep generated benchmark data after the run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset generation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if os.path.isdir(args.data_root) and not args.keep_data:
        shutil.rmtree(args.data_root, ignore_errors=True)
    os.makedirs(args.data_root, exist_ok=True)

    data_model = DataModel(str(args.data_root))
    generate_image_dataset(
        data_model=data_model,
        documents=args.documents,
        pages_per_document=args.pages_per_document,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    engines = select_engines(args.engines or [])
    results: list[BenchmarkResult] = []

    for engine in engines:
        print(f"Running OCR benchmark for engine: {engine}")
        results.append(
            benchmark_engine(
                engine,
                data_model,
                args.documents,
                args.pages_per_document,
                args.width,
                args.height,
            )
        )

    print(format_results(results, args.width, args.height))

    if not args.keep_data:
        shutil.rmtree(args.data_root, ignore_errors=True)

    return 0 if all(result.error is None for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
