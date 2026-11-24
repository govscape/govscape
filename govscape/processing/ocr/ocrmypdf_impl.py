"""OcrMyPDF implementation."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from PIL import Image

from .base_ocr import BaseOCR

try:
    import ocrmypdf
    import pytesseract
except ImportError:
    ocrmypdf = None
    pytesseract = None


class OcrMyPDFImpl(BaseOCR):
    """OCR implementation using Tesseract, through pytesseract.

    Selected as ``ocr_type="ocrmypdf"``, the package it is installed with (the
    ``ocrmypdf`` extra); it doesn't run OcrMyPDF itself.
    """

    # Per-page limits on Tesseract's time, for the first attempt and for a retry
    # at half the resolution.
    TIMEOUT_S = 60
    RETRY_TIMEOUT_S = 30

    def __init__(
        self,
        language: str = "eng",
        output_type: str = "txt",
        max_workers: int | None = None,
    ):
        """Initialize OcrMyPDF.

        Args:
            language: Tesseract language code (e.g., 'eng', 'fra'). Defaults to 'eng'.
            output_type: Output type ('txt' or 'searchable_pdf'). Defaults to 'txt'.
            max_workers: Number of pages to OCR concurrently. Defaults to one per
                CPU; set to 1 to disable concurrency.
        """
        # Tesseract parallelises with OpenMP threads unless OMP_THREAD_LIMIT=1;
        # this engine parallelises across pages instead, which scales better.
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        self.language = language
        self.output_type = output_type
        self.max_workers = max_workers or os.cpu_count() or 1
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate OcrMyPDF installation and dependencies."""
        if ocrmypdf is None:
            raise ImportError(
                "ocrmypdf is not installed. Install it with: pip install ocrmypdf"
            )

        if pytesseract is None:
            raise ImportError(
                "pytesseract is not installed. Install it with: pip install pytesseract"
            )

        self.logger.info(
            f"OcrMyPDF initialized with language: {self.language}, "
            f"output: {self.output_type}"
        )

    def extract_text(self, images: list[np.ndarray]) -> list[str | None]:
        """Extract text from a batch of images using Tesseract.

        Args:
            images: A list of numpy arrays, each a page image.

        Returns:
            A list of extracted text strings, one per input image; None for a
            page that couldn't be OCR'd in time (see TIMEOUT_S) or at all.
        """
        if pytesseract is None:
            self.validate()

        # Each Tesseract call runs in a subprocess (so the GIL is not a
        # bottleneck); OCR the batch's pages concurrently. map()
        # preserves input order, keeping results aligned with pages.
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(self._extract_single, images))

    def _extract_single(self, image: np.ndarray) -> str | None:
        pil_image = (
            Image.fromarray(image.astype("uint8"))
            if isinstance(image, np.ndarray)
            else image
        )
        # Tesseract's time depends on a page's content as much as its size: dense
        # line art (drawings, maps) can take many minutes, stalling the whole
        # batch. Such a page gets one retry at half the resolution.
        try:
            return self._tesseract(pil_image, self.TIMEOUT_S)
        except TimeoutError:
            pass
        try:
            return self._tesseract(pil_image.reduce(2), self.RETRY_TIMEOUT_S)
        except TimeoutError:
            return None

    def _tesseract(self, image: Image.Image, timeout: int) -> str | None:
        """Raises TimeoutError if Tesseract takes longer than timeout seconds."""
        try:
            return pytesseract.image_to_string(
                image, lang=self.language, timeout=timeout
            )
        except RuntimeError as e:
            if "timeout" not in str(e).lower():
                self.logger.error(f"Error during Tesseract text extraction: {e}")
                return None
            self.logger.warning(
                f"Tesseract timed out after {timeout} s on a "
                f"{image.size[0]}x{image.size[1]} page"
            )
            raise TimeoutError from e
        except Exception as e:
            self.logger.error(f"Error during Tesseract text extraction: {e}")
            return None
