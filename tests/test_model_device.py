import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

import numpy as np

import torch

from govscape import text_embedding_models
from govscape.model_device import LazyDeviceModel
from govscape.text_embedding_models import (
    BGE_TextEmbeddingModel,
    Qwen3_TextEmbeddingModel,
)


class FakeSentenceTransformer:
    def __init__(self, model_name, device=None):
        self.load_device = device
        self.calls: list[tuple] = []

    def half(self):
        self.calls.append(("half",))

    def float(self):
        self.calls.append(("float",))

    def to(self, device):
        self.calls.append(("to", device.type))

    def encode(self, text, device, batch_size=None):
        self.calls.append(("encode", device.type))
        return np.zeros(4)


class GPUFullSentenceTransformer(FakeSentenceTransformer):
    def to(self, device):
        super().to(device)
        if device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")


def _cuda_check_forbidden():
    raise AssertionError("CUDA must not be touched while loading the model")


def test_text_model_defers_device_choice_to_first_use(monkeypatch):
    monkeypatch.setattr(
        text_embedding_models, "SentenceTransformer", FakeSentenceTransformer
    )
    monkeypatch.setattr(torch.cuda, "is_available", _cuda_check_forbidden)
    model = BGE_TextEmbeddingModel()
    assert model.model.load_device == "cpu"
    assert model.model.calls == []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model.encode_text("query", is_query=True)
    model.encode_text("another query", is_query=True)

    assert model.device.type == "cuda"
    assert model.model.calls == [
        ("half",),
        ("to", "cuda"),
        ("encode", "cuda"),
        ("encode", "cuda"),
    ]


def test_text_model_stays_full_precision_on_cpu(monkeypatch):
    monkeypatch.setattr(
        text_embedding_models, "SentenceTransformer", FakeSentenceTransformer
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = BGE_TextEmbeddingModel()
    model.encode_text("query")

    assert model.model.calls == [("to", "cpu"), ("encode", "cpu")]


def test_text_model_falls_back_to_cpu_when_gpu_is_full(monkeypatch):
    monkeypatch.setattr(
        text_embedding_models, "SentenceTransformer", GPUFullSentenceTransformer
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = BGE_TextEmbeddingModel()
    model.encode_text("query", is_query=True)
    model.encode_text("another query", is_query=True)

    assert model.device.type == "cpu"
    assert model.model.calls == [
        ("half",),
        ("to", "cuda"),
        ("to", "cpu"),
        ("float",),
        ("encode", "cpu"),
        ("encode", "cpu"),
    ]


class FakeHFModel:
    def __init__(self, calls):
        self.calls = calls

    def set_attn_implementation(self, attn_implementation):
        self.calls.append(("attn", attn_implementation))


class FakeQwen3SentenceTransformer(FakeSentenceTransformer):
    def __getitem__(self, idx):
        return SimpleNamespace(auto_model=FakeHFModel(self.calls))

    def to(self, target):
        if isinstance(target, torch.dtype):
            self.calls.append(("to", target))
        else:
            super().to(target)


class GPUFullQwen3SentenceTransformer(FakeQwen3SentenceTransformer):
    def to(self, target):
        super().to(target)
        if isinstance(target, torch.device) and target.type == "cuda":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")


def test_qwen3_caps_sequence_length(monkeypatch):
    monkeypatch.setattr(
        text_embedding_models, "SentenceTransformer", FakeQwen3SentenceTransformer
    )
    model = Qwen3_TextEmbeddingModel()
    assert model.model.max_seq_length == Qwen3_TextEmbeddingModel.MAX_SEQ_LENGTH


@pytest.mark.parametrize(
    ("capability", "uses_flash"), [((12, 1), True), ((8, 0), True), ((7, 5), False)]
)
def test_qwen3_uses_flash_attention_on_ampere_or_newer_gpus(
    monkeypatch, capability, uses_flash
):
    monkeypatch.setattr(
        text_embedding_models, "SentenceTransformer", FakeQwen3SentenceTransformer
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: capability)
    model = Qwen3_TextEmbeddingModel()
    model.encode_text("query", is_query=True)

    flash = [("attn", Qwen3_TextEmbeddingModel.FLASH_ATTENTION)] if uses_flash else []
    assert model.model.calls == [
        ("to", torch.bfloat16),
        ("to", "cuda"),
        *flash,
        ("encode", "cuda"),
    ]


def test_qwen3_keeps_sdpa_on_cpu(monkeypatch):
    monkeypatch.setattr(
        text_embedding_models, "SentenceTransformer", FakeQwen3SentenceTransformer
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Qwen3_TextEmbeddingModel()
    model.encode_text("query")

    assert model.model.calls == [("to", "cpu"), ("encode", "cpu")]


def test_qwen3_keeps_sdpa_when_gpu_is_full(monkeypatch):
    monkeypatch.setattr(
        text_embedding_models, "SentenceTransformer", GPUFullQwen3SentenceTransformer
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = Qwen3_TextEmbeddingModel()
    model.encode_text("query", is_query=True)

    assert model.device.type == "cpu"
    assert ("attn", Qwen3_TextEmbeddingModel.FLASH_ATTENTION) not in model.model.calls


class HalfThenOOMModel(LazyDeviceModel):
    """Fails the way a real move can: converted to half precision, then out of
    GPU memory part-way through."""

    def __init__(self):
        self.model = torch.nn.Linear(4, 2)

    def _move_model_to(self, device):
        self.model.half()
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")


def test_gpu_fallback_restores_full_precision_cpu_weights(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = HalfThenOOMModel()

    assert model.device.type == "cpu"
    for param in model.model.parameters():
        assert param.dtype == torch.float32
        assert param.device.type == "cpu"


# Mirrors gunicorn's preload_app: models are loaded in a parent process and used
# from a forked child. Runs in a fresh interpreter so CUDA state from other tests
# can't leak in.
_FORK_SCRIPT = textwrap.dedent(
    """
    import multiprocessing as mp
    import sys

    import torch

    import govscape as gs

    text_model = gs.BGE_TextEmbeddingModel()
    visual_model = gs.CLIP_VisualEmbeddingModel()

    def encode(query):
        if len(sys.argv) > 1:
            # Cap this worker's GPU allocations below the size of either model.
            limit = int(sys.argv[1]) * 2**20
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(limit / total)
        text = text_model.encode_text(query, is_query=True)
        visual = visual_model.encode_text(query)
        return text_model.device.type, len(text), visual.shape[-1]

    with mp.get_context("fork").Pool(1) as pool:
        print(pool.apply(encode, ("state budget report",)))
    """
)


def _run_fork_script(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _FORK_SCRIPT, *args],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_models_loaded_before_fork_run_on_gpu_in_child():
    assert _run_fork_script() == "('cuda', 768, 512)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_models_fall_back_to_cpu_in_child_when_gpu_is_full():
    assert _run_fork_script("64") == "('cpu', 768, 512)"
