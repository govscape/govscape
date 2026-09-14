"""Diagnostic test for the front-end search timeout.

Builds a Server with a metadata index that returns many crawl records per
PDF (mimicking a heavily-recrawled page in production), times a real
search() call, and reports the size of `all_records`/`limited_records` that
`Server._build_search_results` computes for each hit.
"""

import time

import pytest

import numpy as np

from govscape.config import ServerConfig
from govscape.query import Query, Response
from govscape.server import Server

MAX_CRAWL_INSTANCES = 50
CRAWL_RECORDS_PER_PDF = 2000
NUM_HITS = 5


class DummyTextModel:
    d = 4

    def encode_text(self, text: str, is_query: bool | None = None) -> np.ndarray:
        return np.ones(self.d, dtype=np.float32)


class DummyVisualModel:
    d = 4

    def encode_text(self, text: str) -> np.ndarray:
        return np.full(self.d, 2.0, dtype=np.float32)


class DummyVectorIndex:
    def __init__(self, *args, **kwargs):
        self._entries = NUM_HITS

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
        pass

    def load_index(self):
        pass

    def search(self, query, k):
        return np.array([]), [], []

    def total_entries(self):
        return 0


class ManyCrawlsMetadataIndex:
    """Every PDF has thousands of crawl records, like a popular gov page."""

    def __init__(self, *args, **kwargs):
        pass

    def load_index(self):
        pass

    def total_entries(self):
        return 1000

    def estimate_selectivity(self, predicates=None):
        return 1.0

    def get_candidate_digests(self, predicates=None):
        return {f"doc_{i}.pdf" for i in range(NUM_HITS)}

    def search(self, pdf_names, predicates=None):
        return {
            name: [
                {
                    "crawl_url": f"https://example.gov/{name}/{i}",
                    "crawl_date": f"20{(i % 24) + 1:02d}-01-01",
                    "sub_domain": "example.gov",
                    "page_count": 2,
                }
                for i in range(CRAWL_RECORDS_PER_PDF)
            ]
            for name in pdf_names
        }


@pytest.fixture()
def server_fixture(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    for sub in (
        "embeddings",
        "embeddings_img_pg",
        "index",
        "index_img_pg",
        "index_keyword",
        "index_metadata",
        "img",
        "metadata",
    ):
        (data_dir / sub).mkdir(parents=True)
    (data_dir / "total_pdfs.txt").write_text(str(NUM_HITS))

    monkeypatch.setattr("govscape.server.FAISSIndex", DummyVectorIndex)
    monkeypatch.setattr("govscape.server.LanceDBKeywordIndex", DummyKeywordIndex)
    monkeypatch.setattr("govscape.server.SQLiteKeywordIndex", DummyKeywordIndex)
    monkeypatch.setattr("govscape.server.WhooshKeywordIndex", DummyKeywordIndex)
    monkeypatch.setattr("govscape.server.SQLiteMetadataIndex", ManyCrawlsMetadataIndex)

    server_config = ServerConfig(
        str(data_dir),
        DummyTextModel(),
        DummyVisualModel(),
        vector_index_type="Memory",
        keyword_index_type="LanceDB",
        k=NUM_HITS,
        max_crawl_instances=MAX_CRAWL_INSTANCES,
    )
    return Server(server_config)


def test_search_runtime_and_crawl_record_lengths(server_fixture):
    server = server_fixture

    start = time.perf_counter()
    response = server.search(Query("test query", search_type="textual"))
    elapsed = time.perf_counter() - start

    print(f"\nserver.search() took {elapsed:.3f}s for {len(response.results)} results")
    assert isinstance(response, Response)
    assert elapsed < 5.0, f"search took too long: {elapsed:.3f}s"

    # Recompute the same all_records/limited_records that
    # Server._build_search_results derives internally, so we can see the
    # counts feeding into each returned result.
    pdf_names = [result["pdf"] for result in response.results]
    pdf_metadata = server.metadata_index.search(pdf_names)
    for name in pdf_names:
        all_records = sorted(
            pdf_metadata[name], key=lambda r: r.get("crawl_date", ""), reverse=True
        )
        limited_records = all_records[: server.max_crawl_instances]
        print(
            f"{name}: len(all_records)={len(all_records)}, "
            f"len(limited_records)={len(limited_records)}"
        )
        assert len(all_records) == CRAWL_RECORDS_PER_PDF
        assert len(limited_records) == min(CRAWL_RECORDS_PER_PDF, MAX_CRAWL_INSTANCES)
