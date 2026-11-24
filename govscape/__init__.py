import importlib
from typing import TYPE_CHECKING

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

__all__ = [
    "BGESmall_TextEmbeddingModel",
    "BGE_TextEmbeddingModel",
    "CLIP_VisualEmbeddingModel",
    "DataModel",
    "Dummy_TextEmbeddingModel",
    "Dummy_VisualEmbeddingModel",
    "FAISSIndex",
    "LanceDBKeywordIndex",
    "LuceneKeywordIndex",
    "PDFProcessingPipeline",
    "Qwen3VLLM_TextEmbeddingModel",
    "Qwen3_TextEmbeddingModel",
    "RemoteDirectoryIterator",
    "SQLiteKeywordIndex",
    "SQLiteMetadataIndex",
    "ST_TextEmbeddingModel",
    "Server",
    "ServerConfig",
    "SigLIP_VisualEmbeddingModel",
    "WhooshKeywordIndex",
    "base_argument_parser",
    "build_data_loader",
    "extract_subdomain",
    "read_txt_file",
    "str2bool",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from .config import DataModel, ServerConfig
    from .data_loader import RemoteDirectoryIterator, build_data_loader
    from .indexing import (
        FAISSIndex,
        LanceDBKeywordIndex,
        LuceneKeywordIndex,
        SQLiteKeywordIndex,
        SQLiteMetadataIndex,
        WhooshKeywordIndex,
    )
    from .pdf_processing_pipeline import PDFProcessingPipeline
    from .server import Server
    from .text_embedding_models import (
        BGE_TextEmbeddingModel,
        BGESmall_TextEmbeddingModel,
        Dummy_TextEmbeddingModel,
        Qwen3_TextEmbeddingModel,
        Qwen3VLLM_TextEmbeddingModel,
        ST_TextEmbeddingModel,
    )
    from .utils import base_argument_parser, extract_subdomain, read_txt_file, str2bool
    from .visual_embedding_models import (
        CLIP_VisualEmbeddingModel,
        Dummy_VisualEmbeddingModel,
        SigLIP_VisualEmbeddingModel,
    )
