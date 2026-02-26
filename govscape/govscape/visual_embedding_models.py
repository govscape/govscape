import math
import multiprocessing as mp
import os
from abc import ABC, abstractmethod

import numpy as np

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer


class VisualEmbeddingModel(ABC):
    @property
    @abstractmethod
    def d(self):
        pass

    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_texts(self, texts):
        pass

    @abstractmethod
    def encode_image(self, jpg_path):
        pass

    @abstractmethod
    def encode_images(self, jpg_paths):
        pass


class Dummy_VisualEmbeddingModel(VisualEmbeddingModel):
    @property
    def d(self):
        return 128

    def __init__(self):
        self._rng = np.random.default_rng()

    def encode_text(self, text):
        return self._rng.random(128).astype(np.float32)

    def encode_texts(self, texts):
        return self._rng.random((len(texts), 128)).astype(np.float32)

    def encode_image(self, jpg_path):
        return self._rng.random(128).astype(np.float32)

    def encode_images(self, jpg_paths):
        return self._rng.random((len(jpg_paths), 128)).astype(np.float32)


class CLIP_VisualEmbeddingModel(VisualEmbeddingModel):
    @property
    def d(self):
        return 512

    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )  # online
        self.model.eval()

    def encode_text(
        self, text
    ):  # note: not doing encode_texts version of this yet because currently not in use.
        # tokenize text
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        tokenized_text = inputs["input_ids"][0]

        # CLIP token limit = 77 so we have to divide into chunks and get
        # embeddings for each of those
        max_chunk_len = 77
        text_chunks = []
        for i in range(0, len(tokenized_text), max_chunk_len):
            if (
                len(tokenized_text[i : i + max_chunk_len]) == max_chunk_len
                or len(text_chunks) == 0
            ):
                text_chunks.append(tokenized_text[i : i + max_chunk_len])

        # stack them all into a single batch so we can compute them all at the same time
        chunk_tensors = [chunk.unsqueeze(0) for chunk in text_chunks]
        batch_input_ids = torch.cat(chunk_tensors, dim=0)
        batch_attention_mask = torch.ones_like(batch_input_ids)

        with torch.no_grad():
            batch_embeddings = self.model.get_text_features(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            )
        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)

        embeddings = batch_embeddings.split(1, dim=0)
        # decision: average embedding to create one embedding per PDF
        return torch.mean(torch.stack(embeddings), dim=0).to("cpu").numpy()

    def encode_texts(self, texts):
        return [self.encode_text(txt) for txt in texts]

    def encode_image(self, jpg_path):
        image = Image.open(jpg_path).convert("RGB")

        # preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)

        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        return image_embedding[0].to("cpu").numpy()

    # single GPU version using self.model (created in __init__)
    @staticmethod
    def _safe_load_image(path, processor):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
            return processor(images=img, return_tensors="pt")["pixel_values"]
        except (OSError, ValueError) as e:
            print(f"Failed to load/process {path}: {e}")
            return None

    # single GPU version using self.model (created in __init__)
    @staticmethod
    def _load_and_process_image(image_paths, processor):
        tensors = []
        for path in image_paths:
            pv = CLIP_VisualEmbeddingModel._safe_load_image(path, processor)
            if pv is not None:
                tensors.append(pv)  # shape [1,3,H,W]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)  # shape [N,3,H,W]

    def encode_images(self, jpg_paths):
        """
        Load and preprocess images in parallel (CPU workers) then run batched
        single-GPU forward passes.
        """
        if not jpg_paths:
            return np.empty((0, self.d), dtype=np.float32)

        cpu_batch_size = 256
        print("Processing Images")
        path_batches = [
            jpg_paths[i : i + cpu_batch_size]
            for i in range(0, len(jpg_paths), cpu_batch_size)
        ]

        # Use spawn to avoid CUDA + fork deadlocks
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=min(os.cpu_count(), len(path_batches))
        ) as pool:
            batch_tensors = pool.starmap(
                self._load_and_process_image,
                [(p, self.processor) for p in path_batches],
            )

        # Flatten and drop empty batches
        image_tensors = []
        for i, bt in enumerate(batch_tensors):
            if bt is not None:
                image_tensors.append(bt)
            else:
                image_tensors.append(
                    np.zeros(len(path_batches[i]), self.d, dtype=np.float32)
                )

        all_pixels = torch.cat(image_tensors, dim=0)
        print(f"Total images preprocessed: {all_pixels.size(0)}")

        all_embeddings = []
        gpu_batch = 256  # GPU forward batch size
        print("Embedding Images")
        for i in range(0, all_pixels.size(0), gpu_batch):
            pixel_batch = all_pixels[i : i + gpu_batch].to(self.device)
            with torch.no_grad():
                embeddings = self.model.get_image_features(pixel_values=pixel_batch)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
            del pixel_batch, embeddings
            torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0).numpy()
