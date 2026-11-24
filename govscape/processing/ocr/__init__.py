"""OCR implementations for text extraction from PDF pages."""

import importlib
from typing import TYPE_CHECKING

# Imported on first use: each engine pulls in its own heavy dependencies (torch,
# transformers, easyocr), and an OCR worker should load only the one it runs.
_EXPORTS = {
    "BaseOCR": "base_ocr",
    "EasyOCRImpl": "easyocr_impl",
    "OLMOcrImpl": "olmocr_impl",
    "OcrMyPDFImpl": "ocrmypdf_impl",
}

__all__ = [
    "BaseOCR",
    "EasyOCRImpl",
    "OLMOcrImpl",
    "OcrMyPDFImpl",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from .base_ocr import BaseOCR
    from .easyocr_impl import EasyOCRImpl
    from .ocrmypdf_impl import OcrMyPDFImpl
    from .olmocr_impl import OLMOcrImpl
