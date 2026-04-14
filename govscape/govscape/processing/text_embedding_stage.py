import logging
import os

import numpy as np

from ..config import DataModel
from ..text_embedding_models import (
    BGE_TextEmbeddingModel,
    BGESmall_TextEmbeddingModel,
    Dummy_TextEmbeddingModel,
    ST_TextEmbeddingModel,
)
from ..utils import read_txt_file
from .processing_stage import ProcessingStage


def _collect_texts_and_paths(data_model: DataModel):
    os.makedirs(data_model.embedding_directory, exist_ok=True)

    text_batch = []
    file_batch = []
    for txt_subdir in os.scandir(data_model.txt_directory):
        if not txt_subdir.is_dir():
            continue
        digest = txt_subdir.name
        embed_dir = data_model.embedding_pdf_directory(digest)
        os.makedirs(embed_dir, exist_ok=True)

        for txt_file in os.listdir(txt_subdir.path):
            full_txt_path = os.path.join(txt_subdir.path, txt_file)
            text = read_txt_file(full_txt_path)
            output_path = os.path.join(embed_dir, txt_file.replace(".txt", ".npy"))
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
    def __init__(self, data_model: DataModel, model_type: str):
        self.data_model = data_model
        self.model = _build_text_model(model_type)

    def validate(self) -> None:
        if not os.path.isdir(self.data_model.txt_directory):
            raise ValueError(
                f"Text input directory does not exist: {self.data_model.txt_directory}"
            )

    def run(self):
        sentences, embed_file_paths = _collect_texts_and_paths(self.data_model)
        embeddings = self.model.encode_text_batch(sentences)
        logging.info(f"Text embeddings computed. Shape: {embeddings.shape}")
        for embedding, output_path in zip(embeddings, embed_file_paths, strict=False):
            np.save(output_path, embedding)
