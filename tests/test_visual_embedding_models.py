from pathlib import Path
from types import SimpleNamespace

import numpy as np

import torch
from PIL import Image

from govscape.visual_embedding_models import (
    CLIP_VisualEmbeddingModel,
    SigLIP_VisualEmbeddingModel,
)


class FakeImageProcessor:
    def __call__(self, images, return_tensors, device):
        return {"pixel_values": torch.stack([image.float() for image in images])}


class FakeModel:
    dtype = torch.float32

    def get_image_features(self, pixel_values):
        means = pixel_values.mean(dim=(2, 3))
        features = torch.cat(
            [means + 1, torch.ones(means.shape[0], 512 - means.shape[1])], dim=1
        )
        return SimpleNamespace(pooler_output=features)


def _fake_clip() -> CLIP_VisualEmbeddingModel:
    model = object.__new__(CLIP_VisualEmbeddingModel)
    model.processor = SimpleNamespace(image_processor=FakeImageProcessor())
    model.model = FakeModel()
    model._device = torch.device("cpu")
    return model


def _image(path: Path, color: tuple[int, int, int]) -> str:
    Image.new("RGB", (4, 4), color).save(path)
    return str(path)


def test_encode_images_keeps_rows_aligned_with_failed_images(tmp_path: Path) -> None:
    red = _image(tmp_path / "red.jpeg", (255, 0, 0))
    blue = _image(tmp_path / "blue.jpeg", (0, 0, 255))
    corrupt = tmp_path / "corrupt.jpeg"
    corrupt.write_bytes(b"not an image")
    paths = [red, str(tmp_path / "missing.jpeg"), blue, str(corrupt), red]

    embeddings = _fake_clip().encode_images(paths)

    assert embeddings.shape == (5, 512)
    assert not embeddings[[1, 3]].any()
    assert embeddings[[0, 2, 4]].any(axis=1).all()
    assert np.allclose(embeddings[0], embeddings[4])
    assert not np.allclose(embeddings[0], embeddings[2])


def test_encode_images_all_failed(tmp_path: Path) -> None:
    embeddings = _fake_clip().encode_images([str(tmp_path / "missing.jpeg")])

    assert embeddings.shape == (1, 512)
    assert not embeddings.any()


class RecordingTokenizer:
    def __init__(self):
        self.texts: list[str] = []

    def __call__(self, text, add_special_tokens):
        self.texts.append(text)
        return {"input_ids": [1, 2, 3]}

    def pad(self, encoded, padding, max_length, return_tensors):
        return {"input_ids": torch.tensor(encoded["input_ids"])}


class FakeTextModel:
    def get_text_features(self, input_ids):
        return SimpleNamespace(pooler_output=torch.ones(input_ids.shape[0], 4))


def test_siglip2_lowercases_query_text():
    for model_name, expected in [
        ("google/siglip2-base-patch16-224", "budget table"),
        ("google/siglip-base-patch16-224", "Budget Table"),
    ]:
        model = object.__new__(SigLIP_VisualEmbeddingModel)
        tokenizer = RecordingTokenizer()
        model.processor = SimpleNamespace(tokenizer=tokenizer)
        model.model = FakeTextModel()
        model._device = torch.device("cpu")
        model.max_text_len = 64
        model.lowercase_text = "siglip2" in model_name
        model.encode_text("Budget Table")
        assert tokenizer.texts == [expected]
