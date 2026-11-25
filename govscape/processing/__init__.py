from .._lazy import lazy_getattr

# Imported on first use, like the top-level package, so worker processes that only
# run PDF extraction don't load the embedding models' dependencies.
_EXPORTS = {
    "OCRProcessingStage": "ocr_processing_stage",
    "PageImageEmbeddingStage": "page_image_embedding_stage",
    "PDFExtractionStage": "pdf_extraction_stage",
    "ProcessingStage": "processing_stage",
    "TextEmbeddingStage": "text_embedding_stage",
}

__all__ = list(_EXPORTS)

__getattr__ = lazy_getattr(__name__, _EXPORTS)
