# AI modified: 2026-04-12 18:24:13 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 18:35:40 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-03-14 21:55:15 1c688b19
# AI modified: 2026-03-15 03:23:45 1c688b19
# AI modified: 2026-03-15 03:26:30 1c688b19
# AI modified: 2026-04-06 00:10:53 434ce298
# AI modified: 2026-04-12 23:51:04 fddc6344a807a84c8b9161bd3ffeded5153c5e27
from pathlib import Path

import pytest

import numpy as np

from govscape.config import IndexConfig, ServerConfig
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
        self.search_calls = 0

    def load_index(self):
        pass

    def search(self, query_vector, k):
        self.search_calls += 1
        distances = np.linspace(0.1, 0.6, self._entries, dtype=np.float32)
        pdf_names = [f"doc_{i}.pdf" for i in range(self._entries)]
        pdf_pages = [str(i) for i in range(self._entries)]
        return distances, pdf_names, pdf_pages

    def total_entries(self):
        return self._entries

    def get_vectors_for_pdf_page_counts(self, pdf_page_counts):
        candidates = {}
        for i in range(self._entries):
            pdf_name = f"doc_{i}.pdf"
            max_pages = pdf_page_counts.get(pdf_name)
            if max_pages is None or int(max_pages) <= 0:
                continue
            candidates[pdf_name] = [
                ("0", np.full((4,), i, dtype=np.float32)),
            ]
        return candidates


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

    def count_filtered_documents(self, filters=None):
        if filters and filters.get("sub_domain") == "narrow.gov":
            return 2
        return 6

    def get_filtered_pdf_page_counts(self, filters=None):
        if filters and filters.get("sub_domain") == "narrow.gov":
            return {"doc_0.pdf": 1, "doc_1.pdf": 1}
        return {f"doc_{i}.pdf": 1 for i in range(6)}

    def get_vectors_for_pdf_page_counts(self, embedding_type, pdf_page_counts):
        _ = embedding_type
        result = {}
        for i in range(6):
            pdf_name = f"doc_{i}.pdf"
            max_pages = pdf_page_counts.get(pdf_name)
            if max_pages is None or int(max_pages) <= 0:
                continue
            result[pdf_name] = [("0", np.full((4,), i, dtype=np.float32))]
        return result


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

    index_config = IndexConfig(
        str(data_dir), vector_index_type="Memory", keyword_index_type="LanceDB"
    )

    text_model = DummyTextModel()
    visual_model = DummyVisualModel()

    server_config = ServerConfig(index_config, text_model, visual_model, k=3)
    return Server(server_config), index_config


@pytest.fixture()
def server_fixture(tmp_path, monkeypatch):
    return _build_server_fixture(tmp_path, monkeypatch)


@pytest.fixture()
def server_fixture_with_blacklist(tmp_path, monkeypatch):
    blacklist_text = "doc_0.pdf\n# takedown ticket-1234\n\n  keyword_doc_0.pdf  \n"
    return _build_server_fixture(tmp_path, monkeypatch, blacklist_text=blacklist_text)


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
    response = server.search(Query("test query", search_type="textual"))

    assert isinstance(response, Response)
    assert len(response.results) == server.config.k


def test_server_visual_search_returns_results(server_fixture):
    server, _ = server_fixture
    response = server.search(Query("visual query", search_type="visual"))

    assert len(response.results) == server.config.k
    assert all(result["pdf"].startswith("doc_") for result in response.results)


def test_server_keyword_search_uses_keyword_index(server_fixture):
    server, _ = server_fixture
    response = server.search(Query("site:gov", search_type="keyword"))

    assert len(response.results) == server.config.k
    assert [result["pdf"] for result in response.results] == [
        "keyword_doc_0.pdf",
        "keyword_doc_1.pdf",
        "keyword_doc_2.pdf",
    ]


def test_vector_search_uses_prefilter_for_selective_filter(server_fixture):
    server, _ = server_fixture
    response = server.search(
        "test query",
        search_type="textual",
        filters={"sub_domain": "narrow.gov"},
    )

    assert len(response["results"]) == 2
    assert response["results"][0]["pdf"] in ["doc_0.pdf", "doc_1.pdf"]


def test_keyword_search_does_not_use_prefilter_branch(server_fixture):
    server, _ = server_fixture
    _ = server.search("site:gov", search_type="keyword", filters={"sub_domain": "x"})

    assert server.keyword_index.last_query == "site:gov"


def test_prefilter_strategy_disabled_when_filtered_documents_zero(server_fixture):
    server, _ = server_fixture
    server.metadata_index.count_filtered_documents = lambda _filters=None: 0

    should_prefilter = server._should_use_prefilter_strategy(
        "textual",
        {"sub_domain": "narrow.gov"},
        server.text_index,
    )

    assert should_prefilter is False


def test_prefilter_vector_path_returns_empty_for_no_candidate_vectors(server_fixture):
    server, _ = server_fixture
    server.metadata_index.get_vectors_for_pdf_page_counts = (
        lambda _embedding_type, _pdf_page_counts: {}
    )

    response = server.search(
        "test query",
        search_type="textual",
        filters={"sub_domain": "narrow.gov"},
    )

    assert response["results"] == []
def test_blacklist_missing_file_is_empty(server_fixture):
    server, _ = server_fixture
    assert server.blacklist == set()


def test_blacklist_loads_from_file(server_fixture_with_blacklist):
    server, _ = server_fixture_with_blacklist
    assert server.blacklist == {"doc_0.pdf", "keyword_doc_0.pdf"}


def test_search_filters_blacklisted_pdfs_textual(server_fixture_with_blacklist):
    server, _ = server_fixture_with_blacklist
    response = server.search(Query("test", search_type="textual"))

    returned = [r["pdf"] for r in response.results]
    assert "doc_0.pdf" not in returned
    assert len(response.results) == server.config.k


def test_search_filters_blacklisted_pdfs_keyword(server_fixture_with_blacklist):
    server, _ = server_fixture_with_blacklist
    response = server.search(Query("site:gov", search_type="keyword"))

    returned = [r["pdf"] for r in response.results]
    assert "keyword_doc_0.pdf" not in returned
    assert returned == [
        "keyword_doc_1.pdf",
        "keyword_doc_2.pdf",
        "keyword_doc_3.pdf",
    ]


def test_pdf_pages_blacklisted_returns_empty_200(server_fixture_with_blacklist):
    server, _ = server_fixture_with_blacklist
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
    server, _ = server_fixture_with_blacklist
    result = server.pdf_pages("doc_1.pdf")

    assert isinstance(result, dict)
    assert "images" in result
