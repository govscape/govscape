import os

import numpy as np

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


class TextEmbeddingStage(ProcessingStage):
    def __init__(self, txts_path, embeddings_path, text_model):
        self.txts_path = txts_path
        self.embeddings_path = embeddings_path
        self.text_model = text_model

    def validate(self) -> list[str]:
        errors = []
        if not os.path.isdir(self.txts_path):
            errors.append(f"Text input directory does not exist: {self.txts_path}")
        return errors

    def run(self):
        sentences, embed_file_paths = _collect_texts_and_paths(
            self.txts_path, self.embeddings_path
        )
        embeddings = self.text_model.encode_text_batch(sentences)
        print("Embeddings computed. Shape:", embeddings.shape)
        for embedding, output_path in zip(embeddings, embed_file_paths, strict=False):
            np.save(output_path, embedding)
