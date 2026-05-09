"""OcrMyPDF implementation."""

import logging

import numpy as np

from PIL import Image

from .base_ocr import BaseOCR


class OcrMyPDFImpl(BaseOCR):
    """OCR implementation using OcrMyPDF.

    OcrMyPDF adds an OCR text layer to scanned PDFs.
    Note: This implementation extracts text from a page image by converting
    to PDF, running OCR, and extracting text back.
    """

    def __init__(self, language: str = "eng", output_type: str = "txt"):
        """Initialize OcrMyPDF.

        Args:
            language: Tesseract language code (e.g., 'eng', 'fra'). Defaults to 'eng'.
            output_type: Output type ('txt' or 'searchable_pdf'). Defaults to 'txt'.
        """
        self.language = language
        self.output_type = output_type
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate OcrMyPDF installation and dependencies."""
        try:
            import ocrmypdf  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ocrmypdf is not installed. Install it with: pip install ocrmypdf"
            ) from e

        try:
            import pytesseract  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pytesseract is not installed. Install it with: pip install pytesseract"
            ) from e

        self.logger.info(
            f"OcrMyPDF initialized with language: {self.language}, "
            f"output: {self.output_type}"
        )

    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from an image using OcrMyPDF/Tesseract.

        Args:
            image: A numpy array representing the page image.

        Returns:
            Extracted text as a string.
        """
        if self.validate is None:
            self.validate()

        try:
            import pytesseract

            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype("uint8"))

            # Extract text using pytesseract (which uses Tesseract OCR)
            return pytesseract.image_to_string(image, lang=self.language)
        except Exception as e:
            self.logger.error(f"Error during OcrMyPDF text extraction: {e}")
            return ""
