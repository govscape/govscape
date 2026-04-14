import logging
import os
from multiprocessing import get_context

import numpy as np

from ..config import DataModel
from ..visual_embedding_models import (
    CLIP_VisualEmbeddingModel,
    Dummy_VisualEmbeddingModel,
)
from .processing_stage import ProcessingStage


def _save_embeddings_batch(embed_and_paths):
    embed, embed_file_paths = embed_and_paths
    for output_path, embedding in zip(embed_file_paths, embed, strict=False):
        np.save(output_path, embedding)


def _build_visual_model(model_type):
    if model_type == "CLIP":
        return CLIP_VisualEmbeddingModel()
    if model_type == "Dummy":
        return Dummy_VisualEmbeddingModel()
    raise ValueError(f"Unsupported visual model type: {model_type}")


class PageImageEmbeddingStage(ProcessingStage):
    def __init__(self, data_model: DataModel, model_type: str, cpu_count: int):
        self.data_model = data_model
        self.model = _build_visual_model(model_type)
        self.cpu_count = cpu_count

    def validate(self) -> None:
        if not os.path.isdir(self.data_model.image_directory):
            raise ValueError(
                f"Image input directory does not exist: \
                            {self.data_model.image_directory}"
            )

    def run(self):
        os.makedirs(self.data_model.embedding_img_pg_directory, exist_ok=True)
        img_paths = []
        embedding_paths = []
        for img_subdir in os.scandir(self.data_model.image_directory):
            if img_subdir.is_dir():
                digest = img_subdir.name
                embed_dir = self.data_model.embedding_img_pg_pdf_directory(digest)
                os.makedirs(embed_dir, exist_ok=True)
                for img_file in os.listdir(img_subdir.path):
                    img_paths.append(os.path.join(img_subdir.path, img_file))
                    embedding_paths.append(
                        os.path.join(embed_dir, os.path.splitext(img_file)[0] + ".npy")
                    )

        logging.info(f"Embedding {len(img_paths)} images")
        emb = self.model.encode_images(img_paths)
        logging.info(f"Image embeddings computed. Shape: {emb.shape}")

        self._save_embeddings_parallel(emb, embedding_paths)

    def _save_embeddings_parallel(self, embed, embed_file_paths):
        chunks = np.array_split(embed, self.cpu_count)
        chunk_embed_file_paths = []
        start = 0
        for chunk in chunks:
            end = chunk.shape[0]
            chunk_embed_file_paths.append(embed_file_paths[start : start + end])
            start = start + end

        if len(chunks) != len(chunk_embed_file_paths):
            raise Exception(
                "chunks and chunk_embed_file_paths should be the same length."
            )

        ctx = get_context("spawn")
        with ctx.Pool(processes=self.cpu_count) as pool:
            pool.map(
                _save_embeddings_batch,
                zip(chunks, chunk_embed_file_paths, strict=False),
            )
