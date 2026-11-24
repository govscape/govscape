import os
import shutil

import pytest

import numpy as np

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


def _build(directory, corpus, start_batch=0, builder=None, crash_after=None):
    """Feed the corpus in 1,000-vector batches, checkpointing when due; stop
    without finishing after crash_after batches."""
    vectors, digests, pages = corpus
    builder = builder or IVFPQIndexBuilder(
        directory,
        nlist=64,
        m=8,
        train_size=3000,
        block_size=1500,
        nprobe=16,
        use_gpu=False,
        checkpoint_every=2500,
    )
    batches = range(0, len(vectors), 1000)
    for batch, start in enumerate(batches):
        if batch < start_batch:
            continue
        if crash_after is not None and batch == crash_after:
            return builder
        end = start + 1000
        builder.add(vectors[start:end], digests[start:end], pages[start:end])
        if builder.checkpoint_due:
            builder.save_checkpoint({"batches_done": batch + 1})
    builder.finish()
    return builder


def test_resumed_build_matches_an_uninterrupted_one(corpus, tmp_path):
    straight = (tmp_path / "straight").as_posix()
    _build(straight, corpus)

    resumed = (tmp_path / "resumed").as_posix()
    # Crash after 8 batches: past the checkpoint at 6,000 vectors, so later
    # blocks, rows and digests were written and must be discarded.
    _build(resumed, corpus, crash_after=8)
    assert IVFPQIndexBuilder.resume((tmp_path / "nothing").as_posix()) is None
    builder, progress = IVFPQIndexBuilder.resume(resumed, use_gpu=False)
    assert progress == {"batches_done": 6} and builder.ntotal == 6000
    _build(resumed, corpus, start_batch=progress["batches_done"], builder=builder)

    assert not [f for f in os.listdir(resumed) if f.startswith("build_")]
    for name in ("vectors.f16", "row_digests.u32", "row_pages.u32", "digests.npy"):
        with (
            open(os.path.join(straight, name), "rb") as a,
            open(os.path.join(resumed, name), "rb") as b,
        ):
            assert a.read() == b.read(), name
    indexes = []
    for directory in (straight, resumed):
        index = IVFPQDiskIndex(directory)
        index.load_index()
        indexes.append(index)
    vectors = corpus[0]
    for query in vectors[::997]:
        a, b = (index.search(query, k=10) for index in indexes)
        assert a[1:] == b[1:]
        np.testing.assert_allclose(a[0], b[0])
