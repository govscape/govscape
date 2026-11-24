"""OCR Processing Stage - Extracts text from PDF pages using OCR engines."""

import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from PIL import Image

from ..config import DataModel
from .ocr.base_ocr import BaseOCR
from .processing_stage import ProcessingStage


def _ocr_engine_class(ocr_type: str) -> type[BaseOCR]:
    # Imported here rather than at module scope so that resolving one engine
    # doesn't pull in every backend's dependencies (torch, tesseract, ...).
    from .ocr import EasyOCRImpl, OcrMyPDFImpl, OLMOcrImpl

    ocr_engines: dict[str, type[BaseOCR]] = {
        "easyocr": EasyOCRImpl,
        "olmocr": OLMOcrImpl,
        "ocrmypdf": OcrMyPDFImpl,
    }

    if ocr_type not in ocr_engines:
        raise ValueError(
            f"Unsupported OCR type: {ocr_type}. "
            f"Must be one of: {list(ocr_engines.keys())}",
        )

    return ocr_engines[ocr_type]


def _build_ocr_engine(ocr_type: str, **kwargs) -> BaseOCR:
    return _ocr_engine_class(ocr_type)(**kwargs)


class OCRProcessingStage(ProcessingStage):
    """Processing stage that OCRs the pages PDFExtractionStage marked for it.

    Reads the high-resolution page renders in
    {ocr_image_directory}/{digest}/{digest}_{pg_no}.png, which PDFExtractionStage
    writes (when given ocr_dpi) only for pages whose PDF text layer is missing or
    nearly empty and that aren't blank. Writes the text to
    {txt_directory}/{digest}/{digest}_{pg_no}.txt, then deletes the renders.
    """

    def __init__(self, data_model: DataModel, ocr_type: str = "easyocr", **ocr_kwargs):
        self.data_model = data_model
        self.ocr_type = ocr_type
        self.ocr_kwargs = dict(ocr_kwargs)
        # Each worker loads a batch's images at once, and a 300 dpi page is ~25 MB
        # decoded, so batches are small.
        self.batch_size = self.ocr_kwargs.pop("batch_size", 32)
        self.max_workers = self.ocr_kwargs.pop("max_workers", None)
        self.logger = logging.getLogger(__name__)
        # Resolve the name now so a typo fails here. The engine itself is only ever
        # instantiated inside the worker processes (see _init_ocr_worker), so the
        # parent never loads an OCR model it wouldn't use.
        _ocr_engine_class(ocr_type)

    def validate(self) -> None:
        return None

    def run(self):
        self.validate()
        pages = self._collect_pages()
        self.logger.info(f"OCR needed for {len(pages)} pages")
        batches = [
            pages[start : start + self.batch_size]
            for start in range(0, len(pages), self.batch_size)
        ]

        processed_count = error_count = 0
        worker_count = self.max_workers or max(1, min(4, os.cpu_count() or 1))
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("spawn"),
            initializer=_init_ocr_worker,
            initargs=(self.ocr_type, self.ocr_kwargs),
        ) as executor:
            results = executor.map(
                _process_batch_in_process, [[p for _, _, p in b] for b in batches]
            )
            for batch, (texts, batch_errors) in zip(batches, results, strict=True):
                error_count += batch_errors
                for (digest, page_num, _), text in zip(batch, texts, strict=True):
                    if self._write_page_text(digest, page_num, text):
                        processed_count += 1
                    else:
                        error_count += 1
        shutil.rmtree(self.data_model.ocr_image_directory, ignore_errors=True)

        self.logger.info(
            f"OCR processing complete. Processed: {processed_count}, "
            f"Errors: {error_count}",
        )

    def _collect_pages(self) -> list[tuple[str, int, str]]:
        """Return (digest, page number, image path) for every page to OCR."""
        pages: list[tuple[str, int, str]] = []
        if not os.path.isdir(self.data_model.ocr_image_directory):
            return pages
        for digest_dir in os.scandir(self.data_model.ocr_image_directory):
            if not digest_dir.is_dir():
                continue
            for page_file in os.listdir(digest_dir.path):
                stem, extension = os.path.splitext(page_file)
                if extension == ".png":
                    page_num = int(stem.rsplit("_", 1)[1])
                    pages.append(
                        (
                            digest_dir.name,
                            page_num,
                            os.path.join(digest_dir.path, page_file),
                        )
                    )
        return sorted(pages)

    def _write_page_text(self, digest: str, page_num: int, text: str) -> bool:
        try:
            txt_output_path = self.data_model.txt_page_path(digest, page_num)
            os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
            with open(txt_output_path, "w", encoding="utf-8") as f:
                f.write(text)
            self.logger.debug(f"Processed: {txt_output_path}")
            return True
        except Exception as e:
            self.logger.error(
                f"Error writing OCR text for {digest} page {page_num}: {e}"
            )
            return False


# Per-worker OCR engine, built once by the process pool initializer and reused
# across every batch that worker handles (avoids reloading the model per batch).
_WORKER_OCR_ENGINE: BaseOCR | None = None


def _init_ocr_worker(ocr_type: str, ocr_kwargs: dict) -> None:
    global _WORKER_OCR_ENGINE
    # Tesseract's OpenMP threads oversubscribe the CPUs when several pages are
    # OCR'd at once; parallelism comes from OCRing pages concurrently instead.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    _WORKER_OCR_ENGINE = _build_ocr_engine(ocr_type, **ocr_kwargs)


def _process_batch_in_process(image_paths: list[str]) -> tuple[list[str], int]:
    logger = logging.getLogger(__name__)
    ocr_engine = _WORKER_OCR_ENGINE
    if ocr_engine is None:
        # Should never happen: the pool initializer builds the engine before any
        # batch is dispatched. Guard defensively so a batch cannot silently vanish.
        logger.error(
            "OCR worker engine was not initialized; failing batch of %d images.",
            len(image_paths),
        )
        return [""] * len(image_paths), len(image_paths)
    try:
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")))
        return ocr_engine.extract_text(images), 0
    except Exception as e:
        logger.error(f"OCR failed for batch with {len(image_paths)} images: {e}")
        return [""] * len(image_paths), len(image_paths)
