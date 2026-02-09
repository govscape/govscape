from pathlib import Path

import pytest

import numpy as np

from govscape.config import IndexConfig, ServerConfig
from govscape.server import Server


class DummyTextModel:
    def __init__(self, d=4):
        self.d = d
        self.last_query = None

    def encode_text(self, text: str, is_query: bool | None = None) -> np.ndarray:
        self.last_query = text
        return np.ones(self.d, dtype=np.float32)


class DummyVisualModel:
    def __init__(self, d=4):
        self.d = d
        self.last_query = None

    def encode_text(self, text: str) -> np.ndarray:
        self.last_query = text
        return np.full(self.d, 2.0, dtype=np.float32)


class DummyVectorIndex:
    def __init__(self, *args, **kwargs):
        self._entries = 6

    def load_index(self):
        pass

    def search(self, query_vector, k):
        distances = np.linspace(0.1, 0.6, self._entries, dtype=np.float32)
        pdf_names = [f"doc_{i}.pdf" for i in range(self._entries)]
        pdf_pages = [str(i) for i in range(self._entries)]
        return distances, pdf_names, pdf_pages

    def total_entries(self):
        return self._entries


class DummyKeywordIndex:
    def __init__(self, *args, **kwargs):
        self._entries = 4
        self.last_query = None
        self.last_k = None

    def load_index(self):
        pass

    def search(self, query, k):
        self.last_query = query
        self.last_k = k
        scores = np.linspace(0.1, 0.4, self._entries, dtype=np.float32)
        pdf_names = [f"keyword_doc_{i}.pdf" for i in range(self._entries)]
        pdf_pages = [str(i) for i in range(self._entries)]
        return scores, pdf_names, pdf_pages

    def total_entries(self):
        return self._entries


class DummyMetadataIndex:
    def __init__(self, *args, **kwargs):
        pass

    def load_index(self):
        pass

    def search(self, pdf_names, filters=None):
        return {
            name: [
                {
                    "crawl_url": "",
                    "crawl_date": "2024-01-01",
                    "sub_domain": "",
                    "page_count": 2,
                }
            ]
            for name in pdf_names
        }


@pytest.fixture()
def server_fixture(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "embeddings").mkdir(parents=True)
    (data_dir / "embeddings_img_pg").mkdir(parents=True)
    (data_dir / "index").mkdir(parents=True)
    (data_dir / "index_img_pg").mkdir(parents=True)
    (data_dir / "index_keyword").mkdir(parents=True)
    (data_dir / "index_metadata").mkdir(parents=True)
    (data_dir / "img").mkdir(parents=True)
    (data_dir / "metadata").mkdir(parents=True)
    (data_dir / "total_pdfs.txt").write_text("0")

    monkeypatch.setattr("govscape.server.FAISSIndex", DummyVectorIndex)
    monkeypatch.setattr("govscape.server.DiskANNIndex", DummyVectorIndex)
    monkeypatch.setattr("govscape.server.LanceDBKeywordIndex", DummyKeywordIndex)
    monkeypatch.setattr("govscape.server.SQLiteKeywordIndex", DummyKeywordIndex)
    monkeypatch.setattr("govscape.server.WhooshKeywordIndex", DummyKeywordIndex)
    monkeypatch.setattr("govscape.server.SQLiteMetadataIndex", DummyMetadataIndex)

    index_config = IndexConfig(
        str(data_dir), vector_index_type="Memory", keyword_index_type="LanceDB"
    )

    text_model = DummyTextModel()
    visual_model = DummyVisualModel()

    server_config = ServerConfig(index_config, text_model, visual_model, k=3)
    return Server(server_config), index_config


def test_server_initialization(server_fixture):
    server, index_config = server_fixture

    assert Path(index_config.index_directory).exists()
    assert Path(index_config.index_keyword_directory).exists()
    assert Path(index_config.index_img_pg_directory).exists()
    assert Path(index_config.image_directory).exists()

    assert server.config.index_directory == index_config.index_directory
    assert server.config.image_directory == index_config.image_directory
    assert server.config.k == 3


def test_server_search_returns_results(server_fixture):
    server, _ = server_fixture
    response = server.search("test query")

    assert isinstance(response, dict)
    assert "results" in response
    assert len(response["results"]) == server.config.k


def test_server_visual_search_returns_results(server_fixture):
    server, _ = server_fixture
    response = server.search("visual query", search_type="visual")

    assert len(response["results"]) == server.config.k
    assert all(result["pdf"].startswith("doc_") for result in response["results"])


def test_server_keyword_search_uses_keyword_index(server_fixture):
    server, _ = server_fixture
    response = server.search("site:gov", search_type="keyword")

    assert len(response["results"]) == server.config.k
    assert [result["pdf"] for result in response["results"]] == [
        "keyword_doc_0.pdf",
        "keyword_doc_1.pdf",
        "keyword_doc_2.pdf",
    ]
