import os
import sqlite3
import struct
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np

FORWARD_INDEX_TYPES = ["SQLite", "LMDB"]

# Open LMDB environments by path; see LMDBForwardIndex._open.
_LMDB_ENVIRONMENTS: dict[str, tuple[Any, int]] = {}
# SQLite connections inherited across fork() are kept referenced but never
# used: closing them in the child (e.g. via garbage collection) can release
# locks held by the parent.
_INHERITED_HANDLES: list[Any] = []


class ForwardIndex(ABC):
    """On-disk mapping from a PDF digest to the vectors of its pages.

    Vector indexes keep one alongside their (often compressed or approximate)
    search structure so that the exact vectors of a candidate set of PDFs can
    be scored directly, which is what hybrid prefiltering needs.

    Database handles are opened lazily, on first read or write, never by
    load_index: the server loads its indexes before gunicorn forks its workers,
    and neither SQLite connections nor LMDB environments may be used across
    fork(). A SQLite connection is transparently reopened in a forked child; an
    LMDB environment opened before fork() cannot be, so using it from a child
    raises an error.
    """

    def __init__(self, index_directory: str):
        self.index_directory = index_directory

    @abstractmethod
    def build_index(self):
        pass

    @abstractmethod
    def add_batch(self, embeddings, digests, pages):
        """Store embeddings (n, d) for the given digests and pages, replacing
        any vector already stored for the same (digest, page)."""

    @abstractmethod
    def load_index(self):
        pass

    @abstractmethod
    def save_index(self):
        pass

    @abstractmethod
    def close(self):
        """Release the underlying database handle."""

    @abstractmethod
    def get_vectors_for_digests(
        self, candidate_digests
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """Return (vectors, digests, pages) for every page of the candidates."""

    @staticmethod
    def _empty_result() -> tuple[np.ndarray, list[str], list[str]]:
        return np.empty((0, 0), dtype=np.float32), [], []


class SQLiteForwardIndex(ForwardIndex):
    def __init__(self, index_directory: str):
        super().__init__(index_directory)
        self.db_path = os.path.join(index_directory, "forward_index.db")
        self.conn: sqlite3.Connection | None = None
        self._conn_pid: int | None = None

    def __getstate__(self):
        # Connections cannot be pickled (e.g. as part of a pickled FAISSIndex).
        return {**self.__dict__, "conn": None, "_conn_pid": None}

    def _connect(self) -> sqlite3.Connection:
        if self.conn is not None and self._conn_pid != os.getpid():
            _INHERITED_HANDLES.append(self.conn)
            self.conn = None
        if self.conn is None:
            os.makedirs(self.index_directory, exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn_pid = os.getpid()
        return self.conn

    def build_index(self):
        self._connect().execute(
            """
            CREATE TABLE IF NOT EXISTS forward_index (
                digest TEXT,
                page TEXT,
                vector BLOB,
                PRIMARY KEY (digest, page)
            ) WITHOUT ROWID;
            """
        )
        self.conn.commit()

    def add_batch(self, embeddings, digests, pages):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        rows = [
            (str(digest), str(page), embedding.tobytes())
            for embedding, digest, page in zip(embeddings, digests, pages, strict=True)
            if digest
        ]
        if not rows:
            return
        self.build_index()
        self.conn.executemany(
            "INSERT OR REPLACE INTO forward_index (digest, page, vector) "
            "VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def load_index(self):
        # The connection is opened lazily on first use (see class docstring).
        return

    def save_index(self):
        if self.conn is not None and self._conn_pid == os.getpid():
            self.conn.commit()

    def close(self):
        if self.conn is not None and self._conn_pid == os.getpid():
            self.conn.close()
        self.conn = None

    def get_vectors_for_digests(self, candidate_digests):
        if not candidate_digests or not os.path.exists(self.db_path):
            return self._empty_result()
        conn = self._connect()

        rows = []
        candidate_list = list(candidate_digests)
        chunk_size = 800
        for i in range(0, len(candidate_list), chunk_size):
            chunk = candidate_list[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows.extend(
                conn.execute(
                    "SELECT digest, page, vector FROM forward_index "
                    f"WHERE digest IN ({placeholders})",
                    chunk,
                ).fetchall()
            )
        if not rows:
            return self._empty_result()

        vectors = np.vstack([np.frombuffer(row[2], dtype=np.float32) for row in rows])
        return vectors, [row[0] for row in rows], [str(row[1]) for row in rows]


class LMDBForwardIndex(ForwardIndex):
    """LMDB-backed forward index storing one record per digest.

    Each value packs all of a PDF's pages so a lookup is a single key read:
    a (num_pages, dim) uint32 header, the float32 vectors, then the page labels
    joined by newlines.
    """

    _HEADER = struct.Struct("<II")

    def __init__(self, index_directory: str, map_size: int = 1 << 40):
        super().__init__(index_directory)
        self.db_path = os.path.join(index_directory, "forward_index.lmdb")
        # map_size only reserves address space; the file grows as data is added.
        self.map_size = map_size
        self.env = None
        self._env_pid: int | None = None

    def __getstate__(self):
        return {**self.__dict__, "env": None, "_env_pid": None}

    def _open(self):
        path = os.path.realpath(self.db_path)
        if self.env is None:
            import lmdb

            # LMDB forbids opening the same environment twice in a process, so
            # instances pointing at the same path share one environment.
            if path not in _LMDB_ENVIRONMENTS:
                os.makedirs(path, exist_ok=True)
                _LMDB_ENVIRONMENTS[path] = (
                    lmdb.open(path, map_size=self.map_size),
                    os.getpid(),
                )
            self.env, self._env_pid = _LMDB_ENVIRONMENTS[path]
        if self._env_pid != os.getpid():
            raise RuntimeError(
                f"LMDB forward index {path} was opened before fork(); it must be "
                "opened lazily in each worker process."
            )
        return self.env

    @classmethod
    def _encode(cls, pages: list[str], vectors: np.ndarray) -> bytes:
        return (
            cls._HEADER.pack(vectors.shape[0], vectors.shape[1])
            + np.ascontiguousarray(vectors, dtype=np.float32).tobytes()
            + "\n".join(pages).encode()
        )

    @classmethod
    def _decode(cls, value: bytes) -> tuple[list[str], np.ndarray]:
        num_pages, dim = cls._HEADER.unpack_from(value)
        vector_end = cls._HEADER.size + num_pages * dim * 4
        vectors = np.frombuffer(
            value, dtype=np.float32, count=num_pages * dim, offset=cls._HEADER.size
        ).reshape(num_pages, dim)
        pages = value[vector_end:].decode().split("\n")
        return pages, vectors

    def build_index(self):
        self._open()

    def add_batch(self, embeddings, digests, pages):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        by_digest: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for embedding, digest, page in zip(embeddings, digests, pages, strict=True):
            if digest:
                by_digest[str(digest)][str(page)] = embedding
        if not by_digest:
            return

        with self._open().begin(write=True) as txn:
            for digest, page_vectors in by_digest.items():
                key = digest.encode()
                existing = txn.get(key)
                if existing is not None:
                    old_pages, old_vectors = self._decode(existing)
                    page_vectors = {
                        **dict(zip(old_pages, old_vectors, strict=True)),
                        **page_vectors,
                    }
                txn.put(
                    key,
                    self._encode(
                        list(page_vectors), np.vstack(list(page_vectors.values()))
                    ),
                )

    def load_index(self):
        # The environment is opened lazily on first use (see ForwardIndex).
        return

    def save_index(self):
        if self.env is not None and self._env_pid == os.getpid():
            self.env.sync()

    def close(self):
        if self.env is not None and self._env_pid == os.getpid():
            self.env.close()
            _LMDB_ENVIRONMENTS.pop(os.path.realpath(self.db_path), None)
        self.env = None
        self._env_pid = None

    def get_vectors_for_digests(self, candidate_digests):
        if not candidate_digests or not os.path.exists(self.db_path):
            return self._empty_result()

        vector_blocks = []
        digests: list[str] = []
        pages: list[str] = []
        with self._open().begin() as txn:
            for digest in candidate_digests:
                value = txn.get(str(digest).encode())
                if value is None:
                    continue
                digest_pages, vectors = self._decode(value)
                vector_blocks.append(vectors)
                digests.extend([digest] * len(digest_pages))
                pages.extend(digest_pages)
        if not vector_blocks:
            return self._empty_result()
        return np.vstack(vector_blocks), digests, pages


def build_forward_index(forward_index_type: str, index_directory: str) -> ForwardIndex:
    if forward_index_type == "SQLite":
        return SQLiteForwardIndex(index_directory)
    if forward_index_type == "LMDB":
        return LMDBForwardIndex(index_directory)
    raise ValueError(
        f"forward_index_type must be one of {FORWARD_INDEX_TYPES}, "
        f"got {forward_index_type!r}"
    )
