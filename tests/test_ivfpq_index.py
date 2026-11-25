import os
import shutil

import pytest

import numpy as np

from govscape.indexing import IVFPQDiskIndex

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
    builder = IVFPQDiskIndex(
        directory,
        nlist=64,
        m=8,
        train_size=5000,
        block_size=3000,
        nprobe=16,
        use_gpu=False,
    )
    for start in range(0, len(vectors), 1000):
        end = start + 1000
        builder.add_batch(vectors[start:end], digests[start:end], pages[start:end])
    builder.save_index()
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
        found = set(zip(found_digests, found_pages, strict=True))
        recalls.append(len(expected & found) / 10)
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


def _add(index, corpus, batches):
    vectors, digests, pages = corpus
    for start in range(1000 * batches.start, 1000 * batches.stop, 1000):
        end = start + 1000
        index.add_batch(vectors[start:end], digests[start:end], pages[start:end])


def _new_index(directory):
    return IVFPQDiskIndex(
        directory,
        nlist=64,
        m=8,
        train_size=3000,
        block_size=1500,
        nprobe=16,
        use_gpu=False,
    )


def test_saved_index_can_be_continued(corpus, tmp_path):
    straight = (tmp_path / "straight").as_posix()
    index = _new_index(straight)
    _add(index, corpus, range(12))
    index.save_index()

    continued = (tmp_path / "continued").as_posix()
    index = _new_index(continued)
    _add(index, corpus, range(6))
    index.save_index()
    # Batches added but never saved, as by a build that stopped: discarded.
    _add(index, corpus, range(6, 8))
    index._flush_block()
    for file in index._files.values():
        file.flush()

    index = _new_index(continued)
    index.load_index()
    assert index.total_entries() == 6000
    _add(index, corpus, range(6, 12))
    index.save_index()

    assert sorted(os.listdir(straight)) == sorted(os.listdir(continued))
    for name in ("vectors.f16", "row_digests.u32", "row_pages.u32", "digests.npy"):
        with (
            open(os.path.join(straight, name), "rb") as a,
            open(os.path.join(continued, name), "rb") as b,
        ):
            assert a.read() == b.read(), name
    indexes = []
    for directory in (straight, continued):
        index = IVFPQDiskIndex(directory)
        index.load_index()
        indexes.append(index)
    assert (
        indexes[0].get_vectors_for_digests(["PDF00150"])[1:]
        == (indexes[1].get_vectors_for_digests(["PDF00150"])[1:])
    )
    vectors = corpus[0]
    for query in vectors[::997]:
        a, b = (index.search(query, k=10) for index in indexes)
        assert a[1:] == b[1:]
        np.testing.assert_allclose(a[0], b[0])
