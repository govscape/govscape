"""OCR Processing Stage - Extracts text from PDF pages using OCR engines."""

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import NamedTuple

import numpy as np

from PIL import Image

from ..config import DataModel
from . import ocr
from .ocr.base_ocr import BaseOCR
from .processing_stage import ProcessingStage

_OCR_ENGINES = {
    "easyocr": "EasyOCRImpl",
    "olmocr": "OLMOcrImpl",
    "ocrmypdf": "OcrMyPDFImpl",
}


def _ocr_engine_class(ocr_type: str) -> type[BaseOCR]:
    if ocr_type not in _OCR_ENGINES:
        raise ValueError(
            f"Unsupported OCR type: {ocr_type}. Must be one of: {list(_OCR_ENGINES)}",
        )
    # Resolved by name so that only the chosen engine's module, and its
    # dependencies, are imported.
    return getattr(ocr, _OCR_ENGINES[ocr_type])


class _Page(NamedTuple):
    digest: str
    page_num: int
    path: str


class OCRProcessingStage(ProcessingStage):
    """Processing stage that OCRs the pages PDFExtractionStage rendered for it.

    With render_ocr_pages, PDFExtractionStage renders each page whose PDF text
    layer is missing or nearly empty, and that isn't blank, at high resolution to
    {ocr_image_directory}/{digest}/{digest}_{pg_no}.png. This stage writes their
    text to {txt_directory}/{digest}/{digest}_{pg_no}.txt, then deletes every
    render, including those of pages that failed.

    Pages are OCR'd in batches of pages_per_batch, each batch in one of
    num_processes worker processes; engine_kwargs go to the engine.
    """

    def __init__(
        self,
        data_model: DataModel,
        ocr_type: str = "easyocr",
        num_processes: int | None = None,
        pages_per_batch: int | None = None,
        **engine_kwargs,
    ):
        cpus = os.cpu_count() or 1
        if ocr_type == "ocrmypdf":
            # One single-threaded Tesseract per CPU, each handed a few pages at a
            # time as it finishes, so no CPU waits on another's slowest page.
            default_processes, default_pages_per_batch = cpus, 4
            engine_kwargs.setdefault("max_workers", 1)
        else:
            # A worker holds its batch's renders in memory, a few MB each.
            default_processes, default_pages_per_batch = min(4, cpus), 32
        self.data_model = data_model
        self.ocr_type = ocr_type
        self.num_processes = num_processes or default_processes
        self.pages_per_batch = pages_per_batch or default_pages_per_batch
        self.engine_kwargs = engine_kwargs
        self.logger = logging.getLogger(__name__)
        # Resolve the name now so a typo fails here. The engine itself is only ever
        # instantiated inside the worker processes (see _init_ocr_worker), so the
        # parent never loads an OCR model it wouldn't use.
        _ocr_engine_class(ocr_type)

    def validate(self) -> None:
        pass  # Nothing to check: no renders just means no page needs OCR.

    def run(self):
        pages: list[_Page] = []
        if os.path.isdir(self.data_model.ocr_image_directory):
            for digest_dir in os.scandir(self.data_model.ocr_image_directory):
                for page_file in os.listdir(digest_dir.path):
                    stem, extension = os.path.splitext(page_file)
                    if extension == ".png":
                        page_num = int(stem.rsplit("_", 1)[1])
                        path = os.path.join(digest_dir.path, page_file)
                        pages.append(_Page(digest_dir.name, page_num, path))
        pages.sort()
        self.logger.info(f"OCR needed for {len(pages)} pages")
        batches = [
            pages[start : start + self.pages_per_batch]
            for start in range(0, len(pages), self.pages_per_batch)
        ]

        processed_count = error_count = 0
        with ProcessPoolExecutor(
            max_workers=self.num_processes,
            mp_context=get_context("spawn"),
            initializer=_init_ocr_worker,
            initargs=(self.ocr_type, self.engine_kwargs),
        ) as executor:
            results = executor.map(
                _process_batch_in_process, [[page.path for page in b] for b in batches]
            )
            for batch, texts in zip(batches, results, strict=True):
                for page, text in zip(batch, texts, strict=True):
                    if text is None:
                        error_count += 1
                        continue
                    txt_path = self.data_model.txt_page_path(page.digest, page.page_num)
                    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    processed_count += 1
        shutil.rmtree(self.data_model.ocr_image_directory, ignore_errors=True)

        self.logger.info(
            f"OCR processing complete. Processed: {processed_count}, "
            f"Errors: {error_count}",
        )


# Per-worker OCR engine, built once by the process pool initializer and reused
# across every batch that worker handles (avoids reloading the model per batch).
_WORKER_OCR_ENGINE: BaseOCR | None = None


def _init_ocr_worker(ocr_type: str, engine_kwargs: dict) -> None:
    global _WORKER_OCR_ENGINE
    _WORKER_OCR_ENGINE = _ocr_engine_class(ocr_type)(**engine_kwargs)


def _process_batch_in_process(image_paths: list[str]) -> list[str | None]:
    """OCR a batch of page renders; None for each page that failed."""
    logger = logging.getLogger(__name__)
    assert _WORKER_OCR_ENGINE is not None  # Built by the pool initializer.
    texts: list[str | None] = [None] * len(image_paths)
    loaded: list[tuple[int, np.ndarray]] = []
    for i, path in enumerate(image_paths):
        try:
            with Image.open(path) as image:
                loaded.append((i, np.asarray(image.convert("L"))))
        except Exception as e:
            logger.error(f"Failed to read OCR render {path}: {e}")
    try:
        results = _WORKER_OCR_ENGINE.extract_text([image for _, image in loaded])
    except Exception as e:
        logger.error(f"OCR failed for batch with {len(loaded)} images: {e}")
        return texts
    for (i, _), text in zip(loaded, results, strict=True):
        texts[i] = text
    return texts
