from ._lazy import lazy_getattr

# Submodules are imported on first use, so that a process which only needs a light
# part of the package (e.g. a PDF extraction worker) doesn't load torch, the models
# and the indexes. Each spawned worker process would otherwise pay ~1 GB for them.
_EXPORTS = {
    "DataModel": "config",
    "ServerConfig": "config",
    "RemoteDirectoryIterator": "data_loader",
    "build_data_loader": "data_loader",
    "FAISSIndex": "indexing",
    "LanceDBKeywordIndex": "indexing",
    "LuceneKeywordIndex": "indexing",
    "SQLiteKeywordIndex": "indexing",
    "SQLiteMetadataIndex": "indexing",
    "TantivyKeywordIndex": "indexing",
    "WhooshKeywordIndex": "indexing",
    "PDFProcessingPipeline": "pdf_processing_pipeline",
    "Server": "server",
    "BGE_TextEmbeddingModel": "text_embedding_models",
    "BGESmall_TextEmbeddingModel": "text_embedding_models",
    "Dummy_TextEmbeddingModel": "text_embedding_models",
    "Qwen3_TextEmbeddingModel": "text_embedding_models",
    "Qwen3VLLM_TextEmbeddingModel": "text_embedding_models",
    "ST_TextEmbeddingModel": "text_embedding_models",
    "base_argument_parser": "utils",
    "extract_subdomain": "utils",
    "read_txt_file": "utils",
    "str2bool": "utils",
    "CLIP_VisualEmbeddingModel": "visual_embedding_models",
    "Dummy_VisualEmbeddingModel": "visual_embedding_models",
    "SigLIP_VisualEmbeddingModel": "visual_embedding_models",
}

__all__ = list(_EXPORTS)

__getattr__ = lazy_getattr(__name__, _EXPORTS)
