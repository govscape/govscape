# global vars
from abc import ABC, abstractmethod

import numpy as np

import torch
from sentence_transformers import SentenceTransformer

GPU_BATCH_SIZE = 16
BATCH_SIZE = 128


class TextEmbeddingModel(ABC):
    @property
    @abstractmethod
    def d(self):
        pass

    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_text_batch(self, texts):
        pass


class ST_TextEmbeddingModel(TextEmbeddingModel):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.d = self.model.get_sentence_embedding_dimension()

    def encode_text(self, text, is_query=False):
        if self.model.device != self.device:
            self.model.to(self.device)
        with torch.no_grad():
            embedding = self.model.encode(
                text, batch_size=GPU_BATCH_SIZE, device=self.device
            )
        return embedding

    def encode_text_batch(self, texts, is_query=False):
        if self.model.device != self.device:
            self.model.to(self.device)
        with torch.no_grad():
            embedding = self.model.encode(
                texts, batch_size=GPU_BATCH_SIZE, device=self.device
            )
        return embedding

    def encode_image(self, jpg_path):
        raise NotImplementedError("TextEmbeddingModel does not support image encoding.")


class BGE_TextEmbeddingModel(TextEmbeddingModel):
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.d = self.model.get_sentence_embedding_dimension()

    def encode_text(self, text, is_query=False):
        if self.model.device != self.device:
            self.model.to(self.device)
        if is_query:
            text = "Represent this sentence for searching relevant passages:" + text
        with torch.no_grad():
            embedding = self.model.encode(
                text, batch_size=GPU_BATCH_SIZE, device=self.device
            )
        return embedding

    def encode_text_batch(self, texts, is_query=False):
        if self.model.device != self.device:
            self.model.to(self.device)
        if is_query:
            texts = [
                "Represent this sentence for searching relevant passages:" + text
                for text in texts
            ]
        with torch.no_grad():
            embedding = self.model.encode(
                texts, batch_size=GPU_BATCH_SIZE, device=self.device
            )
        return embedding

    def encode_image(self, jpg_path):
        raise NotImplementedError("TextEmbeddingModel does not support image encoding.")


class BGESmall_TextEmbeddingModel(TextEmbeddingModel):
    @property
    def d(self):
        return self.model.get_sentence_embedding_dimension()

    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def encode_text(self, text, is_query=False):
        if self.model.device != self.device:
            self.model.to(self.device)
        if is_query:
            text = "Represent this sentence for searching relevant passages:" + text
        with torch.no_grad():
            embedding = self.model.encode(
                text, batch_size=GPU_BATCH_SIZE, device=self.device
            )
        return embedding

    def encode_text_batch(self, texts, is_query=False):
        if self.model.device != self.device:
            self.model.to(self.device)
        if is_query:
            texts = [
                "Represent this sentence for searching relevant passages:" + text
                for text in texts
            ]
        with torch.no_grad():
            embedding = self.model.encode(
                texts, batch_size=GPU_BATCH_SIZE, device=self.device
            )
        return embedding

    def encode_image(self, jpg_path):
        raise NotImplementedError("TextEmbeddingModel does not support image encoding.")


class Dummy_TextEmbeddingModel(TextEmbeddingModel):
    @property
    def d(self):
        return 128

    def __init__(self):
        pass

    def encode_text(self, texts, is_query=False):
        return np.random.rand(self.d)  # Return a random embedding for testing purposes

    def encode_text_batch(self, texts, is_query=False):
        return np.random.rand(
            len(texts), self.d
        )  # Return a random embedding for testing purposes

    def encode_image(self, jpg_path):
        raise NotImplementedError("TextEmbeddingModel does not support image encoding.")
