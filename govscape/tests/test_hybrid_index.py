# AI modified: 2026-04-19 21:12:31 c1b6021e
# AI modified: 2026-04-26 00:00:00 341724af
# AI modified: 2026-04-26 00:00:00 341724af
# AI modified: 2026-04-26T22:00:43Z eac4f332
import numpy as np

from govscape.indexing.hybrid import (
    HybridKeywordMetadataIndex,
    HybridVectorMetadataIndex,
)
from govscape.query import EqualityPredicate


class DummyVectorIndex:
    def __init__(self):
        self.search_calls = []
        self._vectors = {
            "doc_1.pdf": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "doc_2.pdf": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            "doc_3.pdf": np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        }

    def search(self, _query_embedding, _k):
        self.search_calls.append(_k)
        distances = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        names = ["doc_1.pdf", "doc_2.pdf", "doc_3.pdf"]
        pages = ["1", "2", "3"]
        return distances[:_k], names[:_k], pages[:_k]

    def get_vectors_for_digests(self, candidate_digests):
        names = [
            name
            for name in ["doc_1.pdf", "doc_2.pdf", "doc_3.pdf"]
            if name in candidate_digests
        ]
        vectors = np.asarray([self._vectors[name] for name in names], dtype=np.float32)
        pages = [
            "1" if name == "doc_1.pdf" else "2" if name == "doc_2.pdf" else "3"
            for name in names
        ]
        return vectors, names, pages

    def total_entries(self):
        return 3


class SelectiveMetadataIndex:
    def __init__(self):
        self.docs = {"doc_2.pdf"}

    def estimate_selectivity(self, predicates=None):
        return 0.001 if predicates else 1.0

    def total_entries(self):
        return 1000

    def get_candidate_digests(self, _predicates=None):
        return set(self.docs)

    def get_vectors_for_digests(self, _vector_store_key, candidate_digests):
        names = [name for name in ["doc_2.pdf"] if name in candidate_digests]
        vectors = np.asarray([[1.0, 1.0, 1.0, 1.0] for _ in names], dtype=np.float32)
        pages = ["2" for _ in names]
        return vectors, names, pages

    def search(self, pdf_names, _predicates=None):
        return {
            name: [{"crawl_date": "20240101", "crawl_url": "", "sub_domain": ""}]
            for name in pdf_names
            if name == "doc_2.pdf"
        }


class BroadMetadataIndex:
    def __init__(self):
        self.docs = {f"doc_{i}.pdf" for i in range(1000)}

    def estimate_selectivity(self, predicates=None):
        return 0.9 if predicates else 1.0

    def total_entries(self):
        return 1000

    def get_candidate_digests(self, _predicates=None):
        return set(self.docs)

    def get_vectors_for_digests(self, _vector_store_key, candidate_digests):
        ordered = ["doc_1.pdf", "doc_2.pdf", "doc_3.pdf"]
        names = [name for name in ordered if name in candidate_digests]
        vec_map = {
            "doc_1.pdf": [0.0, 0.0, 0.0, 0.0],
            "doc_2.pdf": [1.0, 1.0, 1.0, 1.0],
            "doc_3.pdf": [2.0, 2.0, 2.0, 2.0],
        }
        page_map = {"doc_1.pdf": "1", "doc_2.pdf": "2", "doc_3.pdf": "3"}
        vectors = np.asarray([vec_map[name] for name in names], dtype=np.float32)
        pages = [page_map[name] for name in names]
        return vectors, names, pages

    def search(self, pdf_names, _predicates=None):
        return {
            name: [{"crawl_date": "20240101", "crawl_url": "", "sub_domain": ""}]
            for name in pdf_names
            if name in {"doc_1.pdf", "doc_2.pdf"}
        }


def test_hybrid_uses_prefilter_for_selective_predicates():
    hybrid = HybridVectorMetadataIndex(
        vector_index=DummyVectorIndex(),
        metadata_index=SelectiveMetadataIndex(),
    )

    rows, metadata, state = hybrid.search(
        query_embedding=np.ones(4, dtype=np.float32),
        predicates=[EqualityPredicate("sub_domain", "epa.gov")],
        target_results=2,
    )

    assert state.strategy == "prefilter"
    assert [name for _, name, _ in rows] == ["doc_2.pdf"]
    assert set(metadata.keys()) == {"doc_2.pdf"}


def test_hybrid_uses_postfilter_when_broad():
    hybrid = HybridVectorMetadataIndex(
        vector_index=DummyVectorIndex(),
        metadata_index=BroadMetadataIndex(),
    )

    rows, metadata, state = hybrid.search(
        query_embedding=np.ones(4, dtype=np.float32),
        predicates=[EqualityPredicate("sub_domain", "epa.gov")],
        target_results=2,
    )

    assert state.strategy == "postfilter"
    assert [name for _, name, _ in rows] == ["doc_1.pdf", "doc_2.pdf"]
    assert set(metadata.keys()) == {"doc_1.pdf", "doc_2.pdf"}
    assert hybrid.vector_index.search_calls[0] == 2


class HugeCandidateMetadataIndex(BroadMetadataIndex):
    def __init__(self):
        super().__init__()
        self.docs = {f"doc_{i}.pdf" for i in range(100001)}

    def estimate_selectivity(self, predicates=None):
        return 0.0001 if predicates else 1.0


def test_hybrid_prefilter_cap_forces_postfilter():
    vector_index = DummyVectorIndex()
    hybrid = HybridVectorMetadataIndex(
        vector_index=vector_index,
        metadata_index=HugeCandidateMetadataIndex(),
    )

    _rows, _metadata, state = hybrid.search(
        query_embedding=np.ones(4, dtype=np.float32),
        predicates=[EqualityPredicate("sub_domain", "epa.gov")],
        target_results=2,
    )

    assert state.strategy == "postfilter"
    assert vector_index.search_calls


class DummyKeywordIndex:
    def __init__(self):
        self.search_calls = []
        self.search_filtered_calls = []

    def search(self, _query, k):
        self.search_calls.append(k)
        scores = [3.0, 2.0, 1.0]
        names = ["doc_1.pdf", "doc_2.pdf", "doc_3.pdf"]
        pages = ["1", "2", "3"]
        return scores[:k], names[:k], pages[:k]

    def search_filtered(self, _query, k, allowed_names):
        self.search_filtered_calls.append((k, set(allowed_names)))
        rows = [
            (2.5, "doc_2.pdf", "2"),
            (1.5, "doc_3.pdf", "3"),
        ]
        filtered = [row for row in rows if row[1] in allowed_names]
        return (
            [row[0] for row in filtered][:k],
            [row[1] for row in filtered][:k],
            [row[2] for row in filtered][:k],
        )

    def total_entries(self):
        return 3


class KeywordSelectiveMetadataIndex:
    def estimate_selectivity(self, predicates=None):
        return 0.001 if predicates else 1.0

    def total_entries(self):
        return 1000

    def get_candidate_digests(self, _predicates=None):
        return {"doc_2.pdf"}

    def search(self, pdf_names, _predicates=None):
        return {
            name: [{"crawl_date": "20240101", "crawl_url": "", "sub_domain": ""}]
            for name in pdf_names
            if name == "doc_2.pdf"
        }


def test_keyword_hybrid_prefilter_uses_name_filtered_search():
    keyword_index = DummyKeywordIndex()
    hybrid = HybridKeywordMetadataIndex(
        keyword_index=keyword_index,
        metadata_index=KeywordSelectiveMetadataIndex(),
    )

    rows, metadata, state = hybrid.search(
        query_embedding="resilience",
        predicates=[EqualityPredicate("sub_domain", "epa.gov")],
        target_results=2,
    )

    assert state.strategy == "prefilter"
    assert [name for _, name, _ in rows] == ["doc_2.pdf"]
    assert set(metadata.keys()) == {"doc_2.pdf"}
    assert keyword_index.search_filtered_calls
    assert not keyword_index.search_calls


def test_keyword_hybrid_prefilter_applies_blacklist_before_search_filtered():
    keyword_index = DummyKeywordIndex()
    hybrid = HybridKeywordMetadataIndex(
        keyword_index=keyword_index,
        metadata_index=KeywordSelectiveMetadataIndex(),
    )

    rows, metadata, state = hybrid.search(
        query_embedding="resilience",
        predicates=[EqualityPredicate("sub_domain", "epa.gov")],
        target_results=2,
        blacklist={"doc_2.pdf"},
    )

    assert state.strategy == "prefilter"
    assert rows == []
    assert metadata == {}
    assert keyword_index.search_filtered_calls == []
    assert not keyword_index.search_calls
