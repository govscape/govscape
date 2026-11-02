from pathlib import Path

import numpy as np

import torch
from PIL import Image

from govscape.visual_embedding_models import CLIP_VisualEmbeddingModel


class FakeProcessor:
    def __call__(self, images, return_tensors):
        pixels = torch.tensor(np.asarray(images), dtype=torch.float32)
        return {"pixel_values": pixels.permute(2, 0, 1).unsqueeze(0)}


class FakeModel:
    def get_image_features(self, pixel_values):
        means = pixel_values.mean(dim=(2, 3))
        return torch.cat(
            [means + 1, torch.ones(means.shape[0], 512 - means.shape[1])], dim=1
        )


def _fake_clip() -> CLIP_VisualEmbeddingModel:
    model = object.__new__(CLIP_VisualEmbeddingModel)
    model.processor = FakeProcessor()
    model.model = FakeModel()
    model.device = torch.device("cpu")
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
