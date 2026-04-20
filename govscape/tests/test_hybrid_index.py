# AI modified: 2026-04-19 21:12:31 c1b6021e
import numpy as np

from govscape.indexing.hybrid import HybridVectorMetadataIndex
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

    def get_vectors_for_pdf_names(self, candidate_pdf_names):
        names = [
            name
            for name in ["doc_1.pdf", "doc_2.pdf", "doc_3.pdf"]
            if name in candidate_pdf_names
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
    def estimate_selectivity(self, predicates=None):
        return 0.1 if predicates else 1.0

    def get_candidate_pdf_names(self, _predicates=None):
        return {"doc_2.pdf"}

    def get_vectors_for_pdf_names(self, _vector_store_key, candidate_pdf_names):
        names = [name for name in ["doc_2.pdf"] if name in candidate_pdf_names]
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
    def estimate_selectivity(self, predicates=None):
        return 0.9 if predicates else 1.0

    def get_candidate_pdf_names(self, _predicates=None):
        return {"doc_1.pdf", "doc_2.pdf", "doc_3.pdf"}

    def get_vectors_for_pdf_names(self, _vector_store_key, candidate_pdf_names):
        ordered = ["doc_1.pdf", "doc_2.pdf", "doc_3.pdf"]
        names = [name for name in ordered if name in candidate_pdf_names]
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
        prefilter_threshold=0.2,
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
        prefilter_threshold=0.2,
    )

    rows, metadata, state = hybrid.search(
        query_embedding=np.ones(4, dtype=np.float32),
        predicates=[EqualityPredicate("sub_domain", "epa.gov")],
        target_results=2,
    )

    assert state.strategy == "postfilter"
    assert [name for _, name, _ in rows] == ["doc_1.pdf", "doc_2.pdf"]
    assert set(metadata.keys()) == {"doc_1.pdf", "doc_2.pdf"}
    assert hybrid.vector_index.search_calls[0] == 3
