import os
import shutil

import numpy as np
import pytest

from govscape.indexing.ivfpq import IVFPQDiskIndex, IVFPQIndexBuilder

D = 64
PDFS = 200
PAGES_PER_PDF = 60


@pytest.fixture(scope="module")
def corpus():
    """Clustered unit vectors, like real embeddings: (vectors, digests, pages)."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(50, D))
    vectors = centers[rng.integers(0, 50, PDFS * PAGES_PER_PDF)]
    vectors = vectors + 0.3 * rng.normal(size=vectors.shape)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    digests = [f"PDF{i:05d}" for i in range(PDFS) for _ in range(PAGES_PER_PDF)]
    pages = [p for _ in range(PDFS) for p in range(PAGES_PER_PDF)]
    return vectors.astype(np.float32), digests, pages


@pytest.fixture(scope="module")
def index(corpus, tmp_path_factory):
    vectors, digests, pages = corpus
    directory = tmp_path_factory.mktemp("ivfpq").as_posix()
    # Small blocks, so the build writes several and merges them.
    builder = IVFPQIndexBuilder(
        directory, nlist=64, m=8, train_size=5000, block_size=3000, nprobe=16,
        use_gpu=False,
    )
    for start in range(0, len(vectors), 1000):
        end = start + 1000
        builder.add(vectors[start:end], digests[start:end], pages[start:end])
    builder.finish()
    assert not [f for f in os.listdir(directory) if f.startswith("block_")]

    # Served from a copy: the index must not depend on where it was built.
    served = os.path.join(os.path.dirname(directory), "served")
    shutil.copytree(directory, served)
    index = IVFPQDiskIndex(served)
    index.load_index()
    return index


def test_index_size_and_lookup(index, corpus):
    vectors, digests, pages = corpus
    assert index.total_entries() == len(vectors)
    distances, found_digests, found_pages = index.search(vectors[1234], k=1)
    assert (found_digests[0], found_pages[0]) == (digests[1234], str(pages[1234]))
    assert distances[0] < 1e-4  # float16 rounding only


def test_recall_against_exact_search(index, corpus):
    vectors, digests, pages = corpus
    rng = np.random.default_rng(1)
    queries = rng.normal(size=(50, D)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    recalls = []
    for query in queries:
        exact = np.argsort(((vectors - query) ** 2).sum(axis=1))[:10]
        expected = {(digests[i], str(pages[i])) for i in exact}
        distances, found_digests, found_pages = index.search(query, k=10)
        assert list(distances) == sorted(distances)
        recalls.append(len(expected & set(zip(found_digests, found_pages))) / 10)
    assert np.mean(recalls) >= 0.9


def test_vectors_for_digests(index, corpus):
    vectors, digests, pages = corpus
    found, found_digests, found_pages = index.get_vectors_for_digests(
        ["PDF00007", "PDF00150", "NOT-INDEXED"]
    )
    expected_rows = [i for i, d in enumerate(digests) if d in ("PDF00007", "PDF00150")]
    assert found_digests == [digests[i] for i in expected_rows]
    assert found_pages == [str(pages[i]) for i in expected_rows]
    np.testing.assert_allclose(found, vectors[expected_rows], atol=1e-3)
    empty, no_digests, _ = index.get_vectors_for_digests(["NOT-INDEXED"])
    assert empty.shape == (0, 0) and no_digests == []


def test_too_few_vectors_to_train_all_lists(tmp_path, corpus):
    vectors, digests, pages = corpus
    builder = IVFPQIndexBuilder(
        tmp_path.as_posix(), nlist=1024, m=8, block_size=10**6, use_gpu=False
    )
    builder.add(vectors[:2000], digests[:2000], pages[:2000])
    builder.finish()
    index = IVFPQDiskIndex(tmp_path.as_posix())
    index.load_index()
    assert index.meta["nlist"] == 32  # The largest power of two <= 2000 / 39.
    assert index.search(vectors[5], k=1)[1] == [digests[5]]
