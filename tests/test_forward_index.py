import multiprocessing
import pickle
from pathlib import Path

import pytest

import numpy as np

from govscape.indexing import (
    FAISSIndex,
    ForwardIndex,
    LanceDBVectorIndex,
    build_forward_index,
)

FORWARD_INDEX_TYPES = ["SQLite", "LMDB"]


def _vectors(n: int, d: int = 4, offset: float = 0.0) -> np.ndarray:
    return (np.arange(n * d, dtype=np.float32).reshape(n, d) + offset) / 10


def _as_dict(result) -> dict[tuple[str, str], list[float]]:
    vectors, digests, pages = result
    return {
        (digest, page): vector.tolist()
        for vector, digest, page in zip(vectors, digests, pages, strict=True)
    }


@pytest.fixture(params=FORWARD_INDEX_TYPES)
def forward_index(request: pytest.FixtureRequest, tmp_path: Path) -> ForwardIndex:
    index = build_forward_index(request.param, str(tmp_path / "index"))
    index.build_index()
    return index


def test_get_vectors_for_digests(forward_index: ForwardIndex) -> None:
    vectors = _vectors(3)
    forward_index.add_batch(vectors, ["a", "a", "b"], ["0", "1", "0"])

    result = _as_dict(forward_index.get_vectors_for_digests({"a", "missing"}))

    assert result == {("a", "0"): vectors[0].tolist(), ("a", "1"): vectors[1].tolist()}
    assert forward_index.get_vectors_for_digests(set())[0].shape == (0, 0)
    assert forward_index.get_vectors_for_digests({"missing"})[1] == []


def test_add_batch_replaces_and_extends_pages(forward_index: ForwardIndex) -> None:
    forward_index.add_batch(_vectors(2), ["a", "a"], ["0", "1"])
    replacement = _vectors(2, offset=100)
    forward_index.add_batch(replacement, ["a", "a"], ["1", "2"])

    result = _as_dict(forward_index.get_vectors_for_digests({"a"}))

    assert result == {
        ("a", "0"): _vectors(2)[0].tolist(),
        ("a", "1"): replacement[0].tolist(),
        ("a", "2"): replacement[1].tolist(),
    }


@pytest.mark.parametrize("forward_index_type", FORWARD_INDEX_TYPES)
def test_persists_and_pickles(forward_index_type: str, tmp_path: Path) -> None:
    index_dir = str(tmp_path / "index")
    vectors = _vectors(2)
    index = build_forward_index(forward_index_type, index_dir)
    index.add_batch(vectors, ["a", "b"], ["0", "3"])
    index.save_index()

    reloaded = build_forward_index(forward_index_type, index_dir)
    reloaded.load_index()
    unpickled = pickle.loads(pickle.dumps(index))

    expected = {("a", "0"): vectors[0].tolist(), ("b", "3"): vectors[1].tolist()}
    assert _as_dict(reloaded.get_vectors_for_digests({"a", "b"})) == expected
    assert _as_dict(unpickled.get_vectors_for_digests({"a", "b"})) == expected


@pytest.mark.parametrize("forward_index_type", FORWARD_INDEX_TYPES)
def test_faiss_index_maintains_forward_index(
    forward_index_type: str, tmp_path: Path
) -> None:
    index_dir = str(tmp_path / "faiss")
    vectors = _vectors(3)
    index = FAISSIndex(index_dir, forward_index_type=forward_index_type)
    index.add_batch(vectors, ["a", "b", "b"], ["0", "0", "1"])
    index.save_index()

    reloaded = FAISSIndex(index_dir, forward_index_type=forward_index_type)
    reloaded.load_index()

    assert _as_dict(reloaded.get_vectors_for_digests({"b"})) == {
        ("b", "0"): vectors[1].tolist(),
        ("b", "1"): vectors[2].tolist(),
    }


@pytest.mark.parametrize("forward_index_type", FORWARD_INDEX_TYPES)
def test_lancedb_index_maintains_forward_index(
    forward_index_type: str, tmp_path: Path
) -> None:
    index_dir = str(tmp_path / "lancedb")
    vectors = _vectors(2)
    index = LanceDBVectorIndex(index_dir, forward_index_type=forward_index_type)
    index.add_batch(vectors, ["a", "b"], [0, 1])
    index.save_index()

    reloaded = LanceDBVectorIndex(index_dir, forward_index_type=forward_index_type)
    reloaded.load_index()

    assert _as_dict(reloaded.get_vectors_for_digests({"a", "b"})) == {
        ("a", "0"): vectors[0].tolist(),
        ("b", "1"): vectors[1].tolist(),
    }


def _lookup_in_child(forward_index: ForwardIndex, queue) -> None:
    try:
        queue.put(_as_dict(forward_index.get_vectors_for_digests({"a"})))
    except Exception as exc:
        queue.put(type(exc).__name__)


def _lookup_after_fork(forward_index: ForwardIndex):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_lookup_in_child, args=(forward_index, queue))
    child.start()
    result = queue.get(timeout=30)
    child.join(timeout=30)
    return result


@pytest.mark.parametrize("forward_index_type", FORWARD_INDEX_TYPES)
def test_reads_after_fork(forward_index_type: str, tmp_path: Path) -> None:
    # Mirrors the server with gunicorn's preload_app: the index is loaded in the
    # parent and searched in forked workers.
    index_dir = str(tmp_path / "index")
    vectors = _vectors(1)
    writer = build_forward_index(forward_index_type, index_dir)
    writer.add_batch(vectors, ["a"], ["0"])
    writer.save_index()
    writer.close()

    index = build_forward_index(forward_index_type, index_dir)
    index.load_index()

    expected = {("a", "0"): vectors[0].tolist()}
    assert _lookup_after_fork(index) == expected
    assert _as_dict(index.get_vectors_for_digests({"a"})) == expected


@pytest.mark.parametrize(
    ("forward_index_type", "fails_in_child"), [("SQLite", False), ("LMDB", True)]
)
def test_handle_opened_before_fork(
    forward_index_type: str, fails_in_child: bool, tmp_path: Path
) -> None:
    # SQLite reconnects in the child; LMDB cannot and must fail loudly.
    vectors = _vectors(1)
    index = build_forward_index(forward_index_type, str(tmp_path / "index"))
    index.add_batch(vectors, ["a"], ["0"])
    index.save_index()

    result = _lookup_after_fork(index)
    index.close()

    if fails_in_child:
        assert result == "RuntimeError"
    else:
        assert result == {("a", "0"): vectors[0].tolist()}
