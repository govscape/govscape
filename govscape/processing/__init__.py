import importlib
from typing import TYPE_CHECKING

# Imported on first use, like the top-level package, so worker processes that only
# run PDF extraction don't load the embedding models' dependencies.
_EXPORTS = {
    "OCRProcessingStage": "ocr_processing_stage",
    "PageImageEmbeddingStage": "page_image_embedding_stage",
    "PDFExtractionStage": "pdf_extraction_stage",
    "ProcessingStage": "processing_stage",
    "TextEmbeddingStage": "text_embedding_stage",
}

__all__ = [
    "OCRProcessingStage",
    "PDFExtractionStage",
    "PageImageEmbeddingStage",
    "ProcessingStage",
    "TextEmbeddingStage",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from .ocr_processing_stage import OCRProcessingStage
    from .page_image_embedding_stage import PageImageEmbeddingStage
    from .pdf_extraction_stage import PDFExtractionStage
    from .processing_stage import ProcessingStage
    from .text_embedding_stage import TextEmbeddingStage
