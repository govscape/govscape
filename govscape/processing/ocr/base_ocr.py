"""Abstract base class for OCR implementations."""

from abc import ABC, abstractmethod

import numpy as np


class BaseOCR(ABC):
    """Abstract base class for OCR engines.

    Subclasses should implement text extraction from PDF pages.
    """

    @abstractmethod
    def extract_text(self, images: list[np.ndarray]) -> list[str]:
        """Extract text from a batch of page images.

        Args:
            images: A list of numpy arrays, each a page image: 2-D grayscale (as
                OCRProcessingStage passes them) or 3-D RGB.

        Returns:
            A list of extracted text strings, one per input image and in the
            same order.
        """

    @abstractmethod
    def validate(self) -> None:
        """Validate OCR engine initialization and required dependencies."""
