import os
from multiprocessing import get_context

import numpy as np

from .processing_stage import ProcessingStage


def _save_embeddings_batch(embed_and_paths):
    embed, embed_file_paths = embed_and_paths
    for output_path, embedding in zip(embed_file_paths, embed, strict=False):
        np.save(output_path, embedding)


class PageImageEmbeddingStage(ProcessingStage):
    def __init__(self, img_path, embeddings_img_path, visual_model, cpu_count):
        self.img_path = img_path
        self.embeddings_img_path = embeddings_img_path
        self.visual_model = visual_model
        self.cpu_count = cpu_count

    def validate(self) -> list[str]:
        errors = []
        if not os.path.isdir(self.img_path):
            errors.append(f"Image input directory does not exist: {self.img_path}")
        return errors

    def run(self):
        os.makedirs(self.embeddings_img_path, exist_ok=True)
        img_paths = []
        embedding_paths = []
        for img_subdir in os.scandir(self.img_path):
            if img_subdir.is_dir():
                os.makedirs(
                    os.path.join(self.embeddings_img_path, img_subdir.name),
                    exist_ok=True,
                )
                for img_file in os.listdir(img_subdir.path):
                    img_paths.append(os.path.join(img_subdir.path, img_file))
                    embedding_paths.append(
                        os.path.join(
                            self.embeddings_img_path,
                            img_subdir.name,
                            os.path.splitext(img_file)[0] + ".npy",
                        )
                    )

        print("Embedding this many images: ", len(img_paths))
        emb = self.visual_model.encode_images(img_paths)
        print("Embeddings computed. Shape:", emb.shape)

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
