from pathlib import Path

import numpy as np
import pytest

from govscape.indexing import DiskANNIndex, FAISSIndex


class DummyIndexFlatL2:
	def __init__(self, d):
		self.d = d


class DummyIndexIVFPQ:
	def __init__(self, coarse_quantizer, d, nlist, m, nbits):
		self.d = d
		self.coarse_quantizer = coarse_quantizer
		self.nlist = nlist
		self.m = m
		self.nbits = nbits
		self.nprobe = 0
		self._vectors = np.empty((0, d), dtype=np.float32)

	def train(self, embeddings):
		if embeddings.ndim == 1:
			embeddings = embeddings[np.newaxis, :]
		# No-op for dummy implementation; real FAISS would train here.

	def add(self, embeddings):
		if embeddings.ndim == 1:
			embeddings = embeddings[np.newaxis, :]
		embeddings = embeddings.astype(np.float32)
		self._vectors = np.vstack([self._vectors, embeddings])

	def search(self, query, k):
		if query.ndim == 1:
			query = query[np.newaxis, :]
		distances = np.linalg.norm(self._vectors - query, axis=1)
		order = np.argsort(distances)[:k]
		return (
			distances[order].astype(np.float32)[np.newaxis, :],
			order.astype(np.int64)[np.newaxis, :],
		)

	@property
	def ntotal(self):
		return self._vectors.shape[0]


@pytest.fixture(params=["faiss", "diskann"], ids=["faiss", "diskann"])
def vector_index_case(request, tmp_path, monkeypatch):
	embedding_dim = 4
	embeddings = np.array(
		[
			[0.1, 0.2, 0.3, 0.4],
			[0.4, 0.3, 0.2, 0.1],
			[0.2, 0.1, 0.4, 0.3],
			[0.9, 0.1, 0.1, 0.1],
		],
		dtype=np.float32,
	)
	pdf_names = [
		"energy_policy.pdf",
		"transportation_update.pdf",
		"water_quality.pdf",
		"grid_resilience.pdf",
	]
	pdf_pages = [3, 8, 5, 12]
	query_vector = embeddings[0]
	k_neighbors = 2

	if request.param == "faiss":
		index_dir = tmp_path / "faiss_index"
		monkeypatch.setattr("govscape.indexing.faiss.IndexFlatL2", DummyIndexFlatL2)
		monkeypatch.setattr("govscape.indexing.faiss.IndexIVFPQ", DummyIndexIVFPQ)
		index = FAISSIndex(index_dir.as_posix())

		return {
			"index": index,
			"add_args": (embeddings, pdf_names, pdf_pages),
			"query": query_vector,
			"k": k_neighbors,
			"save": lambda idx: idx.save_index(),
			"reload_factory": lambda: FAISSIndex(index_dir.as_posix()),
			"expected_total": len(pdf_names),
			"pdf_names": tuple(pdf_names),
			"pdf_pages": tuple(pdf_pages),
			"stored_vectors": embeddings,
			"query_for_distance": query_vector,
		}

	# DiskANN setup
	embedding_dir = tmp_path / "diskann_embeddings"
	index_dir = tmp_path / "diskann_index"

	class TestDiskANNIndex(DiskANNIndex):
		def add_batch(self, embeddings, pdf_names, pdf_pages):
			embeddings = np.asarray(embeddings, dtype=np.float32)
			embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
			embedding_path = Path(self.embedding_directory)
			embedding_path.mkdir(parents=True, exist_ok=True)
			bin_path = embedding_path / "embeddings.bin"
			with open(bin_path, "wb") as f:
				np.array([embeddings.shape[0]], dtype=np.int32).tofile(f)
				np.array([embeddings.shape[1]], dtype=np.int32).tofile(f)
				embeddings.tofile(f)
			self.pdf_names = list(pdf_names)
			self.pdf_pages = list(pdf_pages)
			self.page_indices = list(range(len(pdf_names)))

	index = TestDiskANNIndex(embedding_dir.as_posix(), index_dir.as_posix())

	normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
	normalized_query = query_vector / np.linalg.norm(query_vector)

	return {
		"index": index,
		"add_args": (embeddings, pdf_names, pdf_pages),
		"query": query_vector,
		"k": k_neighbors,
		"save": lambda idx: idx.save_index(idx.index_directory),
		"reload_factory": lambda: TestDiskANNIndex(
			embedding_dir.as_posix(), index_dir.as_posix()
		),
		"expected_total": 0,
		"pdf_names": tuple(pdf_names),
		"pdf_pages": tuple(pdf_pages),
		"stored_vectors": normalized_embeddings,
		"query_for_distance": normalized_query,
	}


def test_vector_indices_round_trip(vector_index_case):
	index = vector_index_case["index"]
	embeddings, pdf_names, pdf_pages = vector_index_case["add_args"]
	query = vector_index_case["query"]
	k = vector_index_case["k"]

	index.add_batch(embeddings, pdf_names, pdf_pages)
	index.build_index()
	vector_index_case["save"](index)

	assert index.total_entries() == vector_index_case["expected_total"]

	reloaded_index = vector_index_case["reload_factory"]()
	reloaded_index.load_index()

	search_result = reloaded_index.search(query, k)

	distances = np.asarray(search_result[0]).reshape(-1)
	assert distances.size == k
	assert np.isfinite(distances).all()

	second_component = search_result[1]
	if isinstance(second_component, list):
		indices = [pdf_names.index(name) for name in second_component[:k]]
	else:
		indices = [int(idx) for idx in np.asarray(second_component).reshape(-1)[:k]]

	stored_vectors = vector_index_case["stored_vectors"]
	query_for_distance = vector_index_case["query_for_distance"]
	expected_order = np.argsort(
		np.linalg.norm(stored_vectors - query_for_distance, axis=1)
	)[:k].tolist()
	assert indices == expected_order

	if len(search_result) > 2:
		pages_component = search_result[2]
		expected_pages = [pdf_pages[idx] for idx in indices[: len(pages_component)]]
		assert list(pages_component)[: len(expected_pages)] == expected_pages

	assert reloaded_index.total_entries() == vector_index_case["expected_total"]
