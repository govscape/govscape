import multiprocessing
from pathlib import Path

import pytest

import numpy as np

from govscape.indexing import FAISSIndex, LanceDBVectorIndex


def _vectors(n: int, d: int = 4, offset: float = 0.0) -> np.ndarray:
    return (np.arange(n * d, dtype=np.float32).reshape(n, d) + offset) / 10


def _as_dict(result) -> dict[tuple[str, str], list[float]]:
    vectors, digests, pages = result
    return {
        (digest, page): vector.tolist()
        for vector, digest, page in zip(vectors, digests, pages, strict=True)
    }


@pytest.mark.parametrize("index_class", [FAISSIndex, LanceDBVectorIndex])
def test_vector_indexes_store_page_vectors(index_class, tmp_path: Path) -> None:
    index_dir = str(tmp_path / "index")
    index = index_class(index_dir)
    index.add_batch(_vectors(3), ["a", "a", "b"], ["0", "1", "0"])
    # A later batch replaces page 1 of "a" and adds page 2.
    replacement = _vectors(2, offset=100)
    index.add_batch(replacement, ["a", "a"], ["1", "2"])
    index.save_index()

    reloaded = index_class(index_dir)
    reloaded.load_index()

    assert _as_dict(reloaded.get_vectors_for_digests({"a", "missing"})) == {
        ("a", "0"): _vectors(3)[0].tolist(),
        ("a", "1"): replacement[0].tolist(),
        ("a", "2"): replacement[1].tolist(),
    }
    assert reloaded.get_vectors_for_digests(set())[0].shape == (0, 0)
    assert reloaded.get_vectors_for_digests({"missing"})[1] == []


def _lookup_in_child(index, queue) -> None:
    try:
        queue.put(_as_dict(index.get_vectors_for_digests({"a"})))
    except Exception as exc:
        queue.put(type(exc).__name__)


def _lookup_after_fork(index):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    child = ctx.Process(target=_lookup_in_child, args=(index, queue))
    child.start()
    result = queue.get(timeout=30)
    child.join(timeout=30)
    return result


def test_store_is_opened_in_each_forked_worker(tmp_path: Path) -> None:
    # Mirrors the server with gunicorn's preload_app: the index is loaded in the
    # parent and searched in forked workers.
    index_dir = str(tmp_path / "index")
    writer = FAISSIndex(index_dir)
    writer.add_batch(_vectors(1), ["a"], ["0"])
    writer.save_index()
    writer.page_vectors.close()

    index = FAISSIndex(index_dir)
    index.load_index()
    expected = {("a", "0"): _vectors(1)[0].tolist()}
    assert _lookup_after_fork(index) == expected

    # Once the parent has opened the store, a forked child can't use it, and
    # fails loudly rather than corrupting it.
    assert _as_dict(index.get_vectors_for_digests({"a"})) == expected
    assert _lookup_after_fork(index) == "RuntimeError"
    index.page_vectors.close()
