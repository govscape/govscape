# global vars
from abc import ABC, abstractmethod

import numpy as np

import torch
from sentence_transformers import SentenceTransformer

from .model_device import LazyDeviceModel

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


class _SentenceTransformerModel(LazyDeviceModel, TextEmbeddingModel):
    def __init__(self, model_name):
        # Loaded on the CPU; LazyDeviceModel moves it on first use.
        self.model = SentenceTransformer(model_name, device="cpu")

    @property
    def d(self):
        return self.model.get_sentence_embedding_dimension()

    def _move_model_to(self, device):
        # fp16 is a GPU-only optimization; converting to half precision on CPU
        # is unsupported/unoptimized in PyTorch and makes inference an order
        # of magnitude slower rather than faster.
        if device.type != "cpu":
            self.model.half()
        self.model.to(device)

    def encode_image(self, jpg_path):
        raise NotImplementedError("TextEmbeddingModel does not support image encoding.")


class ST_TextEmbeddingModel(_SentenceTransformerModel):
    def __init__(self):
        super().__init__("sentence-transformers/all-mpnet-base-v2")

    def encode_text(self, text, is_query=False):
        with torch.no_grad():
            return self.model.encode(
                text, batch_size=GPU_BATCH_SIZE, device=self.device
            )

    def encode_text_batch(self, texts, is_query=False):
        with torch.no_grad():
            return self.model.encode(
                texts, batch_size=GPU_BATCH_SIZE, device=self.device
            )


class BGE_TextEmbeddingModel(_SentenceTransformerModel):
    def __init__(self):
        super().__init__("BAAI/bge-base-en-v1.5")

    def encode_text(self, text, is_query=False):
        if is_query:
            text = "Represent this sentence for searching relevant passages:" + text
        with torch.no_grad():
            return self.model.encode(text, batch_size=256, device=self.device)

    def encode_text_batch(self, texts, is_query=False):
        if is_query:
            texts = [
                "Represent this sentence for searching relevant passages:" + text
                for text in texts
            ]
        with torch.no_grad():
            return self.model.encode(
                texts, batch_size=GPU_BATCH_SIZE, device=self.device
            )


class BGESmall_TextEmbeddingModel(_SentenceTransformerModel):
    def __init__(self):
        super().__init__("BAAI/bge-small-en-v1.5")

    def encode_text(self, text, is_query=False):
        if is_query:
            text = "Represent this sentence for searching relevant passages:" + text
        with torch.no_grad():
            return self.model.encode(text, batch_size=512, device=self.device)

    def encode_text_batch(self, texts, is_query=False):
        if is_query:
            texts = [
                "Represent this sentence for searching relevant passages:" + text
                for text in texts
            ]
        with torch.no_grad():
            return self.model.encode(
                texts, batch_size=GPU_BATCH_SIZE, device=self.device
            )


class Qwen3_TextEmbeddingModel(_SentenceTransformerModel):
    QUERY_PREFIX = (
        "Instruct: Given a web search query, retrieve relevant passages that "
        "answer the query\nQuery:"
    )

    # Qwen3 accepts 32k tokens, but the rare very long pages then dominate
    # runtime and GPU memory; few pages exceed this.
    MAX_SEQ_LENGTH = 2048
    FLASH_ATTENTION = "kernels-community/flash-attn2"

    def __init__(self):
        super().__init__("Qwen/Qwen3-Embedding-0.6B")
        self.model.max_seq_length = self.MAX_SEQ_LENGTH

    def _move_model_to(self, device):
        # Qwen3 is trained in bf16 and can overflow in fp16.
        if device.type != "cpu":
            self.model.to(torch.bfloat16)
        self.model.to(device)
        # Prebuilt FlashAttention-2 kernel from the HF Hub (needs Ampere or
        # newer). Switched only once the model is on the GPU, so a CPU
        # fallback keeps SDPA; transformers also falls back to SDPA itself if
        # the kernel can't be loaded.
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
            self.model[0].auto_model.set_attn_implementation(self.FLASH_ATTENTION)

    def encode_text(self, text, is_query=False):
        if is_query:
            text = self.QUERY_PREFIX + text
        with torch.no_grad():
            return self.model.encode(text, device=self.device)

    def encode_text_batch(self, texts, is_query=False):
        if is_query:
            texts = [self.QUERY_PREFIX + text for text in texts]
        with torch.no_grad():
            return self.model.encode(
                texts, batch_size=GPU_BATCH_SIZE, device=self.device
            )


class Dummy_TextEmbeddingModel(TextEmbeddingModel):
    @property
    def d(self):
        return 128

    def __init__(self):
        self._rng = np.random.default_rng()

    def encode_text(self, texts, is_query=False):
        return self._rng.random(
            self.d
        )  # Return a random embedding for testing purposes

    def encode_text_batch(self, texts, is_query=False):
        return self._rng.random(
            (len(texts), self.d)
        )  # Return a random embedding for testing purposes

    def encode_image(self, jpg_path):
        raise NotImplementedError("TextEmbeddingModel does not support image encoding.")
