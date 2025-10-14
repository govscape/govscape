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

		def extra_checks(original_index, reloaded_index, search_result):
			distances, names, pages = search_result
			assert reloaded_index.total_entries() == len(pdf_names)
			assert len(distances) == k_neighbors
			assert names[0] == pdf_names[0]
			assert pages[0] == pdf_pages[0]
			# Ensure distances are finite numbers
			assert np.isfinite(distances).all()

		return {
			"index": index,
			"add_args": (embeddings, pdf_names, pdf_pages),
			"query": query_vector,
			"k": k_neighbors,
			"save": lambda idx: idx.save_index(),
			"reload_factory": lambda: FAISSIndex(index_dir.as_posix()),
			"expected_total": len(pdf_names),
			"extra_checks": extra_checks,
		}

	# DiskANN setup
	embedding_dir = tmp_path / "diskann_embeddings"
	index_dir = tmp_path / "diskann_index"
	call_tracker = {}

	def fake_build_disk_index(**kwargs):
		call_tracker["build_kwargs"] = kwargs

	class FakeStaticDiskIndex:
		def __init__(self, **kwargs):
			call_tracker["static_index_kwargs"] = kwargs

		def search(self, query, k_neighbors, complexity):
			call_tracker["search_kwargs"] = {
				"query": query,
				"k_neighbors": k_neighbors,
				"complexity": complexity,
			}
			internal_indices = np.arange(k_neighbors, dtype=np.int32).reshape(1, -1)
			distances = np.linspace(0.0, 0.25, k_neighbors, dtype=np.float32).reshape(1, -1)
			return internal_indices, distances

	monkeypatch.setattr("govscape.indexing.dap.build_disk_index", fake_build_disk_index)
	monkeypatch.setattr("govscape.indexing.dap.StaticDiskIndex", FakeStaticDiskIndex)

	class TestDiskANNIndex(DiskANNIndex):
		def add_batch(self, embeddings, pdf_names, pdf_pages):
			embeddings = np.asarray(embeddings, dtype=np.float32)
			embedding_path = Path(self.embedding_directory)
			embedding_path.mkdir(parents=True, exist_ok=True)
			(embedding_path / "embeddings.bin").write_bytes(embeddings.tobytes())
			self.pdf_names = list(pdf_names)
			self.pdf_pages = list(pdf_pages)
			self.page_indices = list(range(len(pdf_names)))

	index = TestDiskANNIndex(embedding_dir.as_posix(), index_dir.as_posix())

	def extra_checks(original_index, reloaded_index, search_result):
		assert "build_kwargs" in call_tracker
		assert Path(call_tracker["build_kwargs"]["data"]).name == "embeddings.bin"
		assert call_tracker["build_kwargs"]["index_directory"] == index.index_directory
		assert "static_index_kwargs" in call_tracker
		distances, internal_indices = search_result
		assert distances.shape == (1, k_neighbors)
		assert internal_indices.shape == (1, k_neighbors)
		search_info = call_tracker["search_kwargs"]
		assert search_info["k_neighbors"] == k_neighbors
		assert search_info["complexity"] == k_neighbors * 10
		normalized_query = search_info["query"]
		norm = np.linalg.norm(normalized_query)
		assert np.isclose(norm, 1.0)
		assert reloaded_index.total_entries() == 0

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
		"extra_checks": extra_checks,
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

	vector_index_case["extra_checks"](index, reloaded_index, search_result)
