import logging
from abc import ABC, abstractmethod

import numpy as np

import torch
from PIL import Image
from torchvision.io import ImageReadMode, decode_image, decode_jpeg, read_file
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from .model_device import LazyDeviceModel


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


class CLIP_VisualEmbeddingModel(LazyDeviceModel, VisualEmbeddingModel):
    @property
    def d(self):
        return 512

    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", backend="torchvision"
        )
        # Loaded on the CPU; LazyDeviceModel moves it on first use.
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def _move_model_to(self, device):
        # Half precision on the GPU, as for the text models; it's an order of
        # magnitude slower on the CPU, which keeps float32.
        if device.type != "cpu":
            self.model.half()
        self.model.to(device)

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

        # Since transformers 5 the get_*_features methods return model outputs;
        # pooler_output holds the projected embeddings.
        with torch.no_grad():
            batch_embeddings = self.model.get_text_features(
                input_ids=batch_input_ids, attention_mask=batch_attention_mask
            ).pooler_output
        batch_embeddings = batch_embeddings / batch_embeddings.norm(
            dim=-1, keepdim=True
        )

        embeddings = batch_embeddings.split(1, dim=0)
        # decision: average embedding to create one embedding per PDF
        return torch.mean(torch.stack(embeddings), dim=0).float().cpu().numpy()

    def encode_texts(self, texts):
        return [self.encode_text(txt) for txt in texts]

    def encode_image(self, jpg_path):
        image = Image.open(jpg_path).convert("RGB")

        # preprocess image
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device, self.model.dtype
        )

        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs).pooler_output

        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        return image_embedding[0].float().cpu().numpy()

    # Images per batch for decoding, preprocessing and the forward pass.
    IMAGE_BATCH_SIZE = 256

    def encode_images(self, jpg_paths):
        """Embed page images, one row per path, in order; images that can't be
        read or decoded get a zero vector.

        The JPEGs are decoded (nvJPEG), resized and normalized on the model's
        device, so on a GPU the CPU only reads the files.
        """
        embeddings = np.zeros((len(jpg_paths), self.d), dtype=np.float32)
        failed = 0
        for start in range(0, len(jpg_paths), self.IMAGE_BATCH_SIZE):
            batch_paths = jpg_paths[start : start + self.IMAGE_BATCH_SIZE]
            rows, images = self._decode_images(batch_paths)
            failed += len(batch_paths) - len(rows)
            if not images:
                continue
            pixels = self.processor.image_processor(
                images=images, return_tensors="pt", device=self.device
            )["pixel_values"]
            with torch.no_grad():
                features = self.model.get_image_features(
                    pixel_values=pixels.to(self.device, self.model.dtype)
                ).pooler_output
                features = features / features.norm(dim=-1, keepdim=True)
            embeddings[[start + row for row in rows]] = features.float().cpu().numpy()
        if failed:
            logging.warning(
                "%d of %d images failed to load; using zero embeddings for them",
                failed,
                len(jpg_paths),
            )
        return embeddings

    def _decode_images(self, paths):
        """Return (indices of the decodable images in paths, decoded images)."""
        rows, data = [], []
        for i, path in enumerate(paths):
            try:
                data.append(read_file(path))
                rows.append(i)
            except (OSError, RuntimeError):
                pass
        if not data:
            return [], []
        try:
            return rows, decode_jpeg(data, device=self.device, mode=ImageReadMode.RGB)
        except RuntimeError:
            # One bad image fails the whole batch; decode them one at a time to
            # skip just the bad ones.
            decoded_rows, images = [], []
            for row, image_data in zip(rows, data, strict=True):
                try:
                    image = decode_image(image_data, mode=ImageReadMode.RGB)
                except RuntimeError:
                    continue
                images.append(image.to(self.device))
                decoded_rows.append(row)
            return decoded_rows, images


class SigLIP_VisualEmbeddingModel(CLIP_VisualEmbeddingModel):
    """Google SigLIP image/text embeddings.

    Image encoding reuses the batched CLIP implementation; only model loading
    and text encoding differ.
    """

    DEFAULT_MODEL_NAME = "google/siglip-base-patch16-224"

    @property
    def d(self):
        return self.model.config.vision_config.hidden_size

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(
            model_name, backend="torchvision"
        )
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.max_text_len = self.model.config.text_config.max_position_embeddings

    def encode_text(self, text):
        # SigLIP was trained on text padded to a fixed length, so each chunk is
        # padded to max_text_len rather than masked. Long text is split into
        # chunks whose embeddings are averaged, as for CLIP.
        tokenizer = self.processor.tokenizer
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        chunks = [
            token_ids[i : i + self.max_text_len]
            for i in range(0, max(len(token_ids), 1), self.max_text_len)
        ]
        batch = tokenizer.pad(
            {"input_ids": chunks},
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        with torch.no_grad():
            batch_embeddings = self.model.get_text_features(
                input_ids=batch
            ).pooler_output
        batch_embeddings = batch_embeddings / batch_embeddings.norm(
            dim=-1, keepdim=True
        )
        return torch.mean(batch_embeddings, dim=0, keepdim=True).float().cpu().numpy()
