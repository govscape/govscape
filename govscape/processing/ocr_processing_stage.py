"""OCR Processing Stage - Extracts text from PDF pages using OCR engines."""

import contextlib
import functools
import importlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

from ..config import DataModel
from .ocr.base_ocr import BaseOCR
from .processing_stage import ProcessingStage


@functools.cache
def _load_cv2() -> Any:
    """Return the cv2 module, or None if it is unavailable.

    Imported lazily so that importing this module does not pull in OpenCV.
    """
    try:
        return importlib.import_module("cv2")
    except Exception:
        return None


def _ocr_engine_class(ocr_type: str) -> type[BaseOCR]:
    # Imported here rather than at module scope so that resolving one engine
    # doesn't pull in every backend's dependencies (torch, paddle, ...).
    from .ocr import EasyOCRImpl, OcrMyPDFImpl, OLMOcrImpl, PaddleOCRImpl

    ocr_engines: dict[str, type[BaseOCR]] = {
        "easyocr": EasyOCRImpl,
        "paddleocr": PaddleOCRImpl,
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
    """Processing stage that performs OCR on PDF page images.

    Reads images from {image_directory}/{digest}/{digest}_{pg_no}.jpeg and
    writes extracted text to {txt_directory}/{digest}/{digest}_{pg_no}.txt.

    Only pages that need it are OCR'd: pages whose PDF text layer is missing or
    nearly empty (e.g. a bare page number), and that aren't blank.
    """

    MAX_TEXT_LAYER_CHARS = 20
    # A page is blank if fewer than MIN_INK_FRACTION of its pixels are darker
    # than INK_LEVEL. On a sample of the archive's pages without a text layer,
    # this matched only truly blank pages (~9% of them).
    INK_LEVEL = 160
    MIN_INK_FRACTION = 0.002

    def __init__(self, data_model: DataModel, ocr_type: str = "easyocr", **ocr_kwargs):
        self.data_model = data_model
        self.ocr_type = ocr_type
        self.ocr_kwargs = dict(ocr_kwargs)
        self.batch_size = self.ocr_kwargs.pop("batch_size", 1000)
        self.max_workers = self.ocr_kwargs.pop("max_workers", None)
        self.logger = logging.getLogger(__name__)
        # Resolve the name now so a typo fails here. The engine itself is only ever
        # instantiated inside the worker processes (see _init_ocr_worker), so the
        # parent never loads an OCR model it wouldn't use.
        _ocr_engine_class(ocr_type)

    def validate(self) -> None:
        if _load_cv2() is None:
            raise ImportError(
                "cv2 (OpenCV) is required for OCR processing. "
                "Install it with: pip install opencv-python",
            )

        if not os.path.isdir(self.data_model.image_directory):
            raise ValueError(
                f"Image input directory does not exist: "
                f"{self.data_model.image_directory}",
            )

    def run(self):
        self.validate()
        os.makedirs(self.data_model.txt_directory, exist_ok=True)

        all_images, all_metadata, error_count = self._collect_images_and_metadata()
        batches = [
            all_images[start : start + self.batch_size]
            for start in range(0, len(all_images), self.batch_size)
        ]

        worker_count = self.max_workers or max(1, min(4, os.cpu_count() or 1))
        all_texts: list[str] = []
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("spawn"),
            initializer=_init_ocr_worker,
            initargs=(self.ocr_type, self.ocr_kwargs),
        ) as executor:
            for batch_texts, batch_errors in executor.map(
                _process_batch_in_process, batches
            ):
                all_texts.extend(batch_texts)
                error_count += batch_errors

        processed_count = 0
        for (digest, page_num), text in zip(all_metadata, all_texts, strict=True):
            if self._write_page_text(digest, page_num, text):
                processed_count += 1
            else:
                error_count += 1

        self.logger.info(
            f"OCR processing complete. Processed: {processed_count}, "
            f"Errors: {error_count}",
        )

    def _collect_images_and_metadata(self) -> tuple[list, list[tuple[str, int]], int]:
        cv2 = _load_cv2()
        error_count = 0
        skipped_text = skipped_blank = 0
        all_images: list = []
        all_metadata: list[tuple[str, int]] = []

        for digest_dir in os.scandir(self.data_model.image_directory):
            if not digest_dir.is_dir():
                continue

            digest = digest_dir.name
            os.makedirs(self.data_model.txt_pdf_directory(digest), exist_ok=True)

            page_files = sorted(
                [f for f in os.listdir(digest_dir.path) if f.endswith(".jpeg")],
            )

            for page_file in page_files:
                image_path = os.path.join(digest_dir.path, page_file)
                try:
                    page_num = int(page_file.split("_")[-1].replace(".jpeg", ""))
                    if self._has_text_layer(digest, page_num):
                        skipped_text += 1
                        continue
                    image = cv2.imread(image_path)
                    if image is None:
                        self.logger.warning(f"Failed to read image: {image_path}")
                        error_count += 1
                        continue
                    if self._is_blank(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)):
                        skipped_blank += 1
                        continue
                    if self.ocr_type != "paddleocr":
                        with contextlib.suppress(Exception):
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    all_images.append(image)
                    all_metadata.append((digest, page_num))
                except Exception as e:
                    self.logger.error(f"Error loading {image_path}: {e}")
                    error_count += 1

        self.logger.info(
            f"OCR needed for {len(all_images)} pages; skipped {skipped_text} with a "
            f"text layer and {skipped_blank} blank",
        )
        return all_images, all_metadata, error_count

    def _has_text_layer(self, digest: str, page_num: int) -> bool:
        txt_path = self.data_model.txt_page_path(digest, page_num)
        if not os.path.exists(txt_path):
            return False
        with open(txt_path, encoding="utf-8", errors="replace") as f:
            return len(f.read().strip()) > self.MAX_TEXT_LAYER_CHARS

    def _is_blank(self, gray) -> bool:
        return (gray < self.INK_LEVEL).mean() < self.MIN_INK_FRACTION

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
    _WORKER_OCR_ENGINE = _build_ocr_engine(ocr_type, **ocr_kwargs)


def _process_batch_in_process(batch_images: list) -> tuple[list[str], int]:
    ocr_engine = _WORKER_OCR_ENGINE
    if ocr_engine is None:
        # Should never happen: the pool initializer builds the engine before any
        # batch is dispatched. Guard defensively so a batch cannot silently vanish.
        logging.getLogger(__name__).error(
            "OCR worker engine was not initialized; failing batch of %d images.",
            len(batch_images),
        )
        return [""] * len(batch_images), len(batch_images)
    try:
        texts = ocr_engine.extract_text(batch_images)
        return texts, 0
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"OCR failed for batch with {len(batch_images)} images: {e}")
        return [""] * len(batch_images), len(batch_images)
