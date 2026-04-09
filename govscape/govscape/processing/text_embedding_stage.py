import os

import numpy as np

from ..text_embedding_models import (
    BGE_TextEmbeddingModel,
    BGESmall_TextEmbeddingModel,
    Dummy_TextEmbeddingModel,
    ST_TextEmbeddingModel,
)
from ..utils import read_txt_file
from .processing_stage import ProcessingStage


def _collect_texts_and_paths(txt_path, embed_path):
    os.makedirs(embed_path, exist_ok=True)

    txt_subdirs_paths = [
        txt_subdir.path for txt_subdir in os.scandir(txt_path) if txt_subdir.is_dir()
    ]

    text_batch = []
    file_batch = []
    for txt_subdir_path in txt_subdirs_paths:
        embed_name = os.path.basename(txt_subdir_path)
        embedding_dir = os.path.join(embed_path, embed_name)
        os.makedirs(embedding_dir, exist_ok=True)

        txt_files = os.listdir(txt_subdir_path)
        for txt_file in txt_files:
            full_txt_path = os.path.join(txt_subdir_path, txt_file)
            text = read_txt_file(full_txt_path)
            output_path = os.path.join(embedding_dir, txt_file.replace(".txt", ".npy"))
            text_batch.append(text)
            file_batch.append(output_path)

    return text_batch, file_batch


def _build_text_model(model_type):
    if model_type == "ST":
        return ST_TextEmbeddingModel()
    if model_type == "BGE":
        return BGE_TextEmbeddingModel()
    if model_type == "BGESmall":
        return BGESmall_TextEmbeddingModel()
    if model_type == "Dummy":
        return Dummy_TextEmbeddingModel()
    raise ValueError(f"Unsupported text model type: {model_type}")


class TextEmbeddingStage(ProcessingStage):
    def __init__(self, txts_path, embeddings_path, model_type):
        self.txts_path = txts_path
        self.embeddings_path = embeddings_path
        self.model = _build_text_model(model_type)

    def validate(self) -> None:
        if not os.path.isdir(self.txts_path):
            raise ValueError(f"Text input directory does not exist: {self.txts_path}")

    def run(self):
        sentences, embed_file_paths = _collect_texts_and_paths(
            self.txts_path, self.embeddings_path
        )
        embeddings = self.model.encode_text_batch(sentences)
        print("Embeddings computed. Shape:", embeddings.shape)
        for embedding, output_path in zip(embeddings, embed_file_paths, strict=False):
            np.save(output_path, embedding)
