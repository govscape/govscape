"""OLMOcr OCR implementation."""

import logging

import numpy as np

from .base_ocr import BaseOCR


class OLMOcrImpl(BaseOCR):
    """OCR implementation using OLMOcr.

    OLMOcr is an open-source language model-based OCR system.
    """

    def __init__(self, model_name: str = "default"):
        """Initialize OLMOcr.

        Args:
            model_name: The OLMOcr model to use. Defaults to "default".
        """
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate OLMOcr installation and initialize the model."""
        try:
            import olmocr  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "olmocr is not installed. Install it with: pip install olmocr"
            ) from e

        try:
            # Initialize the model
            from olmocr import OLMOcr as OLMOcrModel

            self.model = OLMOcrModel(model_name=self.model_name)
            self.logger.info(
                f"OLMOcr model '{self.model_name}' initialized successfully"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OLMOcr: {e}") from e

    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from an image using OLMOcr.

        Args:
            image: A numpy array representing the page image.

        Returns:
            Extracted text as a string.
        """
        if self.model is None:
            self.validate()

        try:
            result = self.model.recognize(image)
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            if isinstance(result, str):
                return result
            return str(result)
        except Exception as e:
            self.logger.error(f"Error during OLMOcr text extraction: {e}")
            return ""
