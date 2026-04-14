from pathlib import Path

import pytest

import numpy as np

from govscape.config import ServerConfig
from govscape.query import Query, Response
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

    def search(self, pdf_names, predicates=None):
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


def _build_server_fixture(tmp_path, monkeypatch, blacklist_text: str | None = None):
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
    if blacklist_text is not None:
        (data_dir / "blacklist.txt").write_text(blacklist_text)

    monkeypatch.setattr("govscape.server.FAISSIndex", DummyVectorIndex)
    monkeypatch.setattr("govscape.server.LanceDBKeywordIndex", DummyKeywordIndex)
    monkeypatch.setattr("govscape.server.SQLiteKeywordIndex", DummyKeywordIndex)
    monkeypatch.setattr("govscape.server.WhooshKeywordIndex", DummyKeywordIndex)
    monkeypatch.setattr("govscape.server.SQLiteMetadataIndex", DummyMetadataIndex)

    text_model = DummyTextModel()
    visual_model = DummyVisualModel()

    server_config = ServerConfig(
        str(data_dir),
        text_model,
        visual_model,
        vector_index_type="Memory",
        keyword_index_type="LanceDB",
        k=3,
    )
    return Server(server_config)


@pytest.fixture()
def server_fixture(tmp_path, monkeypatch):
    return _build_server_fixture(tmp_path, monkeypatch)


@pytest.fixture()
def server_fixture_with_blacklist(tmp_path, monkeypatch):
    blacklist_text = "doc_0.pdf\n# takedown ticket-1234\n\n  keyword_doc_0.pdf  \n"
    return _build_server_fixture(tmp_path, monkeypatch, blacklist_text=blacklist_text)


def test_server_initialization(server_fixture):
    server = server_fixture
    data_model = server.config.data_model

    assert Path(data_model.index_directory).exists()
    assert Path(data_model.index_keyword_directory).exists()
    assert Path(data_model.index_img_pg_directory).exists()
    assert Path(data_model.image_directory).exists()

    assert server.data_model.index_directory == data_model.index_directory
    assert server.data_model.image_directory == data_model.image_directory
    assert server.config.k == 3


def test_server_search_returns_results(server_fixture):
    server = server_fixture
    response = server.search(Query("test query", search_type="textual"))

    assert isinstance(response, Response)
    assert len(response.results) == server.config.k


def test_server_visual_search_returns_results(server_fixture):
    server = server_fixture
    response = server.search(Query("visual query", search_type="visual"))

    assert len(response.results) == server.config.k
    assert all(result["pdf"].startswith("doc_") for result in response.results)


def test_server_keyword_search_uses_keyword_index(server_fixture):
    server = server_fixture
    response = server.search(Query("site:gov", search_type="keyword"))

    assert len(response.results) == server.config.k
    assert [result["pdf"] for result in response.results] == [
        "keyword_doc_0.pdf",
        "keyword_doc_1.pdf",
        "keyword_doc_2.pdf",
    ]


def test_blacklist_missing_file_is_empty(server_fixture):
    server = server_fixture
    assert server.blacklist == set()


def test_blacklist_loads_from_file(server_fixture_with_blacklist):
    server = server_fixture_with_blacklist
    assert server.blacklist == {"doc_0.pdf", "keyword_doc_0.pdf"}


def test_search_filters_blacklisted_pdfs_textual(server_fixture_with_blacklist):
    server = server_fixture_with_blacklist
    response = server.search(Query("test", search_type="textual"))

    returned = [r["pdf"] for r in response.results]
    assert "doc_0.pdf" not in returned
    assert len(response.results) == server.config.k


def test_search_filters_blacklisted_pdfs_keyword(server_fixture_with_blacklist):
    server = server_fixture_with_blacklist
    response = server.search(Query("site:gov", search_type="keyword"))

    returned = [r["pdf"] for r in response.results]
    assert "keyword_doc_0.pdf" not in returned
    assert returned == [
        "keyword_doc_1.pdf",
        "keyword_doc_2.pdf",
        "keyword_doc_3.pdf",
    ]


def test_pdf_pages_blacklisted_returns_empty_200(server_fixture_with_blacklist):
    server = server_fixture_with_blacklist
    result = server.pdf_pages("doc_0.pdf")

    assert result == {
        "images": [],
        "crawl_url": "",
        "crawl_date": "",
        "sub_domain": "",
        "has_more_crawls": False,
        "crawl_instances": [],
    }


def test_pdf_pages_non_blacklisted_unaffected(server_fixture_with_blacklist):
    server = server_fixture_with_blacklist
    result = server.pdf_pages("doc_1.pdf")

    assert isinstance(result, dict)
    assert "images" in result
