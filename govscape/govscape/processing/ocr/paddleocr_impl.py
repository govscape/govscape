"""PaddleOCR implementation."""

import logging

import numpy as np

from .base_ocr import BaseOCR

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


class PaddleOCRImpl(BaseOCR):
    """OCR implementation using PaddleOCR.

    PaddleOCR is a multilingual OCR toolkit with high accuracy and efficiency.
    """

    def __init__(self, language: str = "en", use_gpu: bool = False):
        """Initialize PaddleOCR.

        Args:
            language: Language code (e.g., 'en', 'ch', 'fr'). Defaults to 'en'.
            use_gpu: Whether to use GPU for inference. Defaults to False.
        """
        self.language = language
        self.use_gpu = use_gpu
        self.ocr = None
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate PaddleOCR installation and initialize the OCR engine."""
        if PaddleOCR is None:
            raise ImportError(
                "paddleocr is not installed. Install it with: "
                "pip install paddleocr paddlepaddle"
            )

        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True, lang=self.language, use_gpu=self.use_gpu
            )
            self.logger.info(
                f"PaddleOCR initialized with language: {self.language}, "
                f"GPU: {self.use_gpu}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {e}") from e

    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from an image using PaddleOCR.

        Args:
            image: A numpy array representing the page image.

        Returns:
            Extracted text as a string.
        """
        if self.ocr is None:
            self.validate()

        try:
            result = self.ocr.ocr(image, cls=True)
            text_lines = []
            if result:
                for line in result:
                    for detection in line:
                        text = detection[1][0]
                        text_lines.append(text)
            return "\n".join(text_lines)
        except Exception as e:
            self.logger.error(f"Error during PaddleOCR text extraction: {e}")
            return ""
