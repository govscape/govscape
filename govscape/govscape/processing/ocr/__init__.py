"""
OCR module for govscape processing.

This module contains abstract base classes and concrete implementations
for various OCR engines used in document processing.
"""

from .abstract_ocr import AbstractOCR
from .olm_ocr import OlmOCR
from .easy_ocr import EasyOCR
from .paddle_my_ocr import PaddleMyOCR
from .ocr_my_pdf import OCRMyPDF

__all__ = [
    'AbstractOCR',
    'OlmOCR', 
    'EasyOCR',
    'PaddleMyOCR',
    'OCRMyPDF'
]