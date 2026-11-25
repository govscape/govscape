"""OCR implementations for text extraction from PDF pages."""

from ..._lazy import lazy_getattr

# Imported on first use: each engine pulls in its own heavy dependencies (torch,
# transformers, easyocr), and an OCR worker should load only the one it runs.
_EXPORTS = {
    "BaseOCR": "base_ocr",
    "EasyOCRImpl": "easyocr_impl",
    "OLMOcrImpl": "olmocr_impl",
    "OcrMyPDFImpl": "ocrmypdf_impl",
}

__all__ = list(_EXPORTS)

__getattr__ = lazy_getattr(__name__, _EXPORTS)
