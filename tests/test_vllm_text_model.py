import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import numpy as np

from govscape.text_embedding_models import Qwen3VLLM_TextEmbeddingModel


class FakeVLLM:
    """Answers /health (unhealthy for the first few checks, like a server that's
    still loading) and /v1/embeddings, embedding each text as [len(text), 0]."""

    def __init__(self, unhealthy_checks=0):
        self.requests: list[dict] = []
        self.unhealthy_checks = unhealthy_checks
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                healthy = fake.unhealthy_checks <= 0
                fake.unhealthy_checks -= 1
                self.send_response(200 if self.path == "/health" and healthy else 503)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                # Out of order, to check the client sorts by index.
                data = [
                    {"index": i, "embedding": [float(len(text)), 0.0]}
                    for i, text in enumerate(body["input"])
                ][::-1]
                payload = json.dumps({"data": data}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def fake_vllm():
    fake = FakeVLLM()
    yield fake
    fake.server.shutdown()


def test_embeds_batches_in_order_across_concurrent_requests(fake_vllm, monkeypatch):
    monkeypatch.setattr(Qwen3VLLM_TextEmbeddingModel, "TEXTS_PER_REQUEST", 3)
    model = Qwen3VLLM_TextEmbeddingModel(url=fake_vllm.url)
    texts = ["x" * n for n in range(1, 11)]

    embeddings = model.encode_text_batch(texts)

    assert embeddings.dtype == np.float32
    assert embeddings[:, 0].tolist() == list(range(1, 11))
    assert sorted(len(r["input"]) for r in fake_vllm.requests) == [1, 3, 3, 3]
    for request in fake_vllm.requests:
        assert request["model"] == "Qwen/Qwen3-Embedding-0.6B"
        assert request["truncation_side"] == "right"


def test_queries_get_the_instruction_prefix(fake_vllm):
    model = Qwen3VLLM_TextEmbeddingModel(url=fake_vllm.url)
    model.encode_text("snow removal", is_query=True)
    model.encode_text("snow removal")

    assert [r["input"] for r in fake_vllm.requests] == [
        [Qwen3VLLM_TextEmbeddingModel.QUERY_PREFIX + "snow removal"],
        ["snow removal"],
    ]


def test_waits_for_the_server_to_become_healthy(monkeypatch):
    fake = FakeVLLM(unhealthy_checks=2)
    monkeypatch.setattr("govscape.text_embedding_models.time.sleep", lambda s: None)
    try:
        model = Qwen3VLLM_TextEmbeddingModel(url=fake.url)
        assert model.encode_text_batch(["abc"])[:, 0].tolist() == [3.0]
        assert fake.unhealthy_checks < 0
    finally:
        fake.server.shutdown()


def test_empty_batch_makes_no_requests(fake_vllm):
    model = Qwen3VLLM_TextEmbeddingModel(url=fake_vllm.url)
    assert model.encode_text_batch([]).shape == (0, 1024)
    assert fake_vllm.requests == []
