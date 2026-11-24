"""OCR implementations for text extraction from PDF pages."""

from typing import TYPE_CHECKING

from ..._lazy import lazy_getattr

# Imported on first use: each engine pulls in its own heavy dependencies (torch,
# transformers, easyocr), and an OCR worker should load only the one it runs.
_EXPORTS = {
    "BaseOCR": "base_ocr",
    "EasyOCRImpl": "easyocr_impl",
    "OLMOcrImpl": "olmocr_impl",
    "OcrMyPDFImpl": "ocrmypdf_impl",
}

if TYPE_CHECKING:
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

__getattr__ = lazy_getattr(__name__, _EXPORTS)
