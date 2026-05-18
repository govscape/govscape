"""EasyOCR implementation."""

import logging

import numpy as np

from .base_ocr import BaseOCR

try:
    import easyocr
except ImportError:
    easyocr = None


class EasyOCRImpl(BaseOCR):
    """OCR implementation using EasyOCR.

    EasyOCR is a Python library for OCR supporting 80+ languages.
    """

    def __init__(self, languages: list = None, gpu: bool = False):
        """Initialize EasyOCR.

        Args:
            languages: List of language codes (e.g., ['en', 'fr']). Defaults to ['en'].
            gpu: Whether to use GPU for inference. Defaults to False.
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.reader = None
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate EasyOCR installation and initialize the reader."""
        if easyocr is None:
            raise ImportError(
                "easyocr is not installed. Install it with: pip install easyocr"
            )

        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            self.logger.info(
                f"EasyOCR reader initialized with languages: {self.languages}, "
                f"GPU: {self.gpu}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR: {e}") from e

    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from an image using EasyOCR.

        Args:
            image: A numpy array representing the page image.

        Returns:
            Extracted text as a string.
        """
        if self.reader is None:
            self.validate()

        try:
            results = self.reader.readtext(image)
            text_lines = [detection[1] for detection in results]
            return "\n".join(text_lines)
        except Exception as e:
            self.logger.error(f"Error during EasyOCR text extraction: {e}")
            return ""
