"""OCR implementations for text extraction from PDF pages."""

from .base_ocr import BaseOCR
from .easyocr_impl import EasyOCRImpl
from .ocrmypdf_impl import OcrMyPDFImpl
from .olmocr_impl import OLMOcrImpl

__all__ = [
    "BaseOCR",
    "EasyOCRImpl",
    "OLMOcrImpl",
    "OcrMyPDFImpl",
]
