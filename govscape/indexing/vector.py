import contextlib
import logging
import os
import pickle as pkl
import shutil
import struct
import sys
import types
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np

import pyarrow as pa

from .base import AbstractIndex
from .kv_store import KVStore


# Avoid annoying output from faiss during import
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


with suppress_output():
    import faiss


class AbstractVectorIndex(AbstractIndex, ABC):
    """A vector index of page vectors, built with build_index(), add_batch() for
    every batch, then save_index(), and served after load_index()."""

    @abstractmethod
    def build_index(self):
        pass

    @abstractmethod
    def add_batch(self, embeddings, pdf_names, pdf_pages):
        pass

    @abstractmethod
    def load_index(self):
        pass

    @abstractmethod
    def save_index(self):
        pass

    @abstractmethod
    def search(self, query_vector, k):
        """
        Search for the k closest PDFs to the query vector.
        :param query_vector: The vector to search for.
        :param k: The number of closest arrays to return.
        :return: A tuple of distances, pdf_names, and pages.
        """

    @abstractmethod
    def total_entries(self):
        """
        Returns the total number of embeddings in the index.
        :return: Total number of embeddings.
        """

    @abstractmethod
    def get_vectors_for_digests(self, candidate_digests):
        """Return (vectors, digests, pages) for every page of the candidates."""


# Each PDF's page vectors, by digest, for get_vectors_for_digests().
PAGE_VECTORS_DB = "forward_index.lmdb"
# A value: (pages, d) as uint32, the float32 vectors, then the page labels joined
# by newlines.
_PAGE_VECTORS_HEADER = struct.Struct("<II")


def _decode_page_vectors(value: bytes) -> tuple[list[str], np.ndarray]:
    n, d = _PAGE_VECTORS_HEADER.unpack_from(value)
    start = _PAGE_VECTORS_HEADER.size
    vectors = np.frombuffer(value, np.float32, n * d, start).reshape(n, d)
    return value[start + n * d * 4 :].decode().split("\n"), vectors


def add_page_vectors(store: KVStore, embeddings, digests, pages) -> None:
    """Add each page's vector to its PDF's, replacing any already stored for it."""
    by_digest: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for embedding, digest, page in zip(
        np.asarray(embeddings, dtype=np.float32).reshape(len(digests), -1),
        digests,
        pages,
        strict=True,
    ):
        if digest:
            by_digest[str(digest)][str(page)] = embedding
    keys = [digest.encode() for digest in by_digest]
    items = []
    for key, existing, page_vectors in zip(
        keys, store.get_many(keys), by_digest.values(), strict=True
    ):
        if existing is not None:
            old_pages, old_vectors = _decode_page_vectors(existing)
            page_vectors = dict(zip(old_pages, old_vectors, strict=True)) | page_vectors
        vectors = np.vstack(list(page_vectors.values()))
        items.append(
            (
                key,
                _PAGE_VECTORS_HEADER.pack(*vectors.shape)
                + vectors.tobytes()
                + "\n".join(page_vectors).encode(),
            )
        )
    store.put_many(items)


def get_page_vectors(store: KVStore, candidate_digests):
    """(vectors, digests, pages) for every page of the candidates."""
    if not candidate_digests or not store.exists():
        return np.empty((0, 0), dtype=np.float32), [], []
    candidates = [str(digest) for digest in candidate_digests]
    blocks, digests, pages = [], [], []
    values = store.get_many(digest.encode() for digest in candidates)
    for digest, value in zip(candidates, values, strict=True):
        if value is not None:
            digest_pages, vectors = _decode_page_vectors(value)
            blocks.append(vectors)
            digests += [digest] * len(digest_pages)
            pages += digest_pages
    if not blocks:
        return np.empty((0, 0), dtype=np.float32), [], []
    return np.vstack(blocks), digests, pages


class _FAISSUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        # Indexes saved before page vectors moved to a KVStore pickled the
        # forward index class it replaced; load_index() doesn't use it.
        if module == "govscape.indexing.forward":
            return types.SimpleNamespace
        return super().find_class(module, name)


class FAISSIndex(AbstractVectorIndex):
    def __init__(self, index_directory, index_type="IVFPQ"):
        self.index_directory = index_directory
        self.page_vectors = KVStore(os.path.join(index_directory, PAGE_VECTORS_DB))
        self.faiss_index = None
        self.index_type = index_type
        self.d = None
        self.is_trained = False
        self.train_batch = None
        self.pdf_names = []
        self.pdf_pages = []
        faiss.omp_set_num_threads(os.cpu_count())

    def add_batch(self, embeddings, pdf_names, pdf_pages):
        # embeddings: list or array of shape (n, d)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        if self.d is None:
            self.d = embeddings.shape[1]
            self.train_batch = np.array([], dtype=np.float32).reshape(0, self.d)

        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatL2(self.d)

        if not self.is_trained:
            self.train_batch = np.vstack((self.train_batch, embeddings))
            if self.train_batch.shape[0] >= 8192 * 39:
                if self.index_type == "IVFPQ":
                    coarse_quantizer = faiss.IndexFlatL2(self.d)
                    self.faiss_index = faiss.IndexIVFPQ(
                        coarse_quantizer, self.d, 8192, int(self.d / 4), 8
                    )
                elif self.index_type == "Flat":
                    self.faiss_index = faiss.IndexFlatL2(self.d)
                elif self.index_type == "IVF":
                    quantizer = faiss.IndexFlatL2(
                        self.d
                    )  # coarse quantizer used for IVF (inverted file) indexing
                    self.faiss_index = faiss.IndexIVFFlat(
                        quantizer, self.d, 8192, faiss.METRIC_L2
                    )
                elif self.index_type == "HNSW":
                    self.faiss_index = faiss.IndexHNSWFlat(self.d, 32)
                    self.faiss_index.hnsw.efConstruction = 200
                    self.faiss_index.hnsw.efSearch = 200
                else:
                    raise ValueError(f"Unsupported index type: {self.index_type}")
                self.faiss_index.train(self.train_batch)
                self.faiss_index.nprobe = 32
                self.faiss_index.add(
                    self.train_batch[: -embeddings.shape[0], :]
                )  # add all but the last batch
                self.is_trained = True

        if embeddings.shape[1] != self.d:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.d}, got {embeddings.shape[1]}"
            )
        self.faiss_index.add(embeddings)
        self.pdf_names.extend(pdf_names)
        self.pdf_pages.extend(pdf_pages)
        add_page_vectors(self.page_vectors, embeddings, pdf_names, pdf_pages)

    def build_index(self):
        return

    def save_index(self):
        os.makedirs(self.index_directory, exist_ok=True)
        index_path = os.path.join(self.index_directory, "faiss_index.pkl")
        with open(index_path, "wb") as handle:
            pkl.dump(self, handle)
        self.page_vectors.sync()
        print(f"Index saved to {self.index_directory}/faiss_index.pkl")
        return

    def load_index(self):
        index_path = os.path.join(self.index_directory, "faiss_index.pkl")
        if not os.path.exists(index_path):
            return
        with open(index_path, "rb") as handle:
            index = _FAISSUnpickler(handle).load()
        self.faiss_index = index.faiss_index
        self.train_batch = index.train_batch
        self.is_trained = index.is_trained
        self.pdf_names = index.pdf_names
        self.pdf_pages = index.pdf_pages
        self.d = index.faiss_index.d
        print(f"Index loaded from {self.index_directory}/faiss_index.pkl")
        return

    def search(self, query_embedding, k):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        if query_embedding.ndim != 2:
            raise ValueError("Query embedding must be a 2D array.")
        distances, indices = self.faiss_index.search(query_embedding, k)
        distances = distances[0]
        indices = indices[0]
        name_results = []
        page_results = []
        for i in range(indices.shape[0]):
            # parse file information for page
            pdf_name = self.pdf_names[indices[i]]
            pdf_page = self.pdf_pages[indices[i]]
            name_results.append(pdf_name)
            page_results.append(pdf_page)
        return distances, name_results, page_results

    def total_entries(self):
        return self.faiss_index.ntotal

    def get_vectors_for_digests(self, candidate_digests):
        return get_page_vectors(self.page_vectors, candidate_digests)


class LanceDBVectorIndex(AbstractVectorIndex):
    def __init__(self, index_directory, table_name="vector_index"):
        self.index_directory = index_directory
        self.page_vectors = KVStore(os.path.join(index_directory, PAGE_VECTORS_DB))
        self.table_name = table_name
        self.db = None
        self.table = None
        self.d = None

    def _connect(self):
        if self.db is None:
            # Imported lazily: importing lancedb starts a background asyncio event
            # loop, which makes forking (used by the data/processing pools) unsafe.
            from lancedb import connect

            os.makedirs(self.index_directory, exist_ok=True)
            self.db = connect(self.index_directory)

    def build_index(self):
        self._connect()
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            return

        if self.d is None:
            raise ValueError(
                "Vector dimension must be known before creating LanceDB vector table"
            )

        if self.table_name not in self.db.table_names():
            schema = pa.schema(
                [
                    pa.field("vector", pa.list_(pa.float32(), self.d)),
                    pa.field("pdf_name", pa.string()),
                    pa.field("page", pa.int32()),
                ]
            )
            self.table = self.db.create_table(self.table_name, schema=schema)

    def add_batch(self, embeddings, pdf_names, pdf_pages):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if self.d is None:
            self.d = embeddings.shape[1]
        elif embeddings.shape[1] != self.d:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.d}, got {embeddings.shape[1]}"
            )

        if self.table is None:
            self.build_index()

        rows = [
            {
                "vector": embeddings[i].tolist(),
                "pdf_name": str(pdf_name),
                "page": int(page),
            }
            for i, (pdf_name, page) in enumerate(
                zip(pdf_names, pdf_pages, strict=False)
            )
        ]
        if rows:
            self.table.add(rows)
        add_page_vectors(self.page_vectors, embeddings, pdf_names, pdf_pages)

    def save_index(self):
        self.page_vectors.sync()
        if self.table is None:
            self.load_index()
        if self.table is None:
            return

        if self.table.count_rows() > 1024:
            try:
                self.table.create_index(vector_column_name="vector")
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise
        self.table.optimize()

    def load_index(self):
        self._connect()
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            self.build_index()
        if self.table is not None:
            try:
                self.d = int(self.table.schema.field("vector").type.list_size)
            except Exception:
                # Keep d unknown on reload if schema does not expose fixed vector size.
                self.d = None

    def search(self, query_vector, k):
        if self.table is None:
            self.load_index()

        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 2:
            if query_vector.shape[0] != 1:
                raise ValueError("Query embedding must be a 1D vector or shape (1, d)")
            query_vector = query_vector[0]
        if query_vector.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector")
        if self.d is not None and query_vector.shape[0] != self.d:
            raise ValueError(
                "Query dimension mismatch: "
                f"expected {self.d}, got {query_vector.shape[0]}"
            )

        results = (
            self.table.search(query_vector.tolist(), vector_column_name="vector")
            .limit(int(k))
            .to_list()
        )
        distances = [float(r.get("_distance", 0.0)) for r in results]
        pdf_names = [r["pdf_name"] for r in results]
        pages = [int(r["page"]) for r in results]
        return distances, pdf_names, pages

    def total_entries(self):
        if self.table is None:
            self.load_index()
        return self.table.count_rows()

    def get_vectors_for_digests(self, candidate_digests):
        return get_page_vectors(self.page_vectors, candidate_digests)


IVFPQ_INDEX_FILE = "ivfpq.index"
IVFPQ_DATA_FILE = "ivfpq.ivfdata"
VECTORS_FILE = "vectors.f16"
ROW_DIGESTS_FILE = "row_digests.u32"
ROW_PAGES_FILE = "row_pages.u32"
DIGEST_ROWS_DB = "digest_rows.lmdb"
# FAISS recommends at least this many training points per coarse centroid.
MIN_TRAIN_POINTS_PER_LIST = 39


class IVFPQDiskIndex(AbstractVectorIndex):
    """A disk-based IVF-PQ index for archive-scale serving, served with almost
    everything memory-mapped from disk:

    - ivfpq.index, ivfpq.ivfdata: a FAISS IndexIVFPQ whose inverted lists
      (each vector's PQ code and row id) are read from disk at search time; only
      the coarse centroids are held in memory.
    - vectors.f16: every vector in float16, by row, to rerank PQ candidates
      exactly and to return all of a PDF's page vectors for prefiltering.
    - row_digests.u32, row_pages.u32: each row's PDF (an index into digests.npy)
      and page number.
    - digests.npy: the PDF digests, by id.
    - digest_rows.lmdb: each PDF's rows, by digest (a KVStore).

    Building: the first train_size vectors passed to add_batch() are buffered to
    train the index, so they should be a fair sample (digests are hashes, so any
    shard is). FAISS trains the coarse quantizer and assigns vectors to lists on
    the GPU when it has one; PQ encoding is done on the CPU. Encoded vectors are
    written in blocks of block_size, which save_index() merges with the lists
    already saved. A saved index can be added to after load_index(), so a build
    can be saved and continued; rows written after the last save (by a build that
    stopped) are discarded.
    """

    # PQ candidates per result, reranked with the exact (float16) vectors. On the
    # 156k-page pilot (2,048 lists), 1,000 candidates and nprobe 128 found 98% of
    # the exact top 10, against 86% with 100 candidates.
    REFINE_FACTOR = 50
    MIN_CANDIDATES = 1000

    def __init__(
        self,
        index_directory: str,
        nlist: int = 131072,
        m: int = 64,
        train_size: int | None = None,
        block_size: int = 20_000_000,
        nprobe: int = 128,
        use_gpu: bool | None = None,
    ):
        self.index_directory = index_directory
        self.nlist = nlist
        self.m = m
        self.train_size = train_size or 50 * nlist
        self.block_size = block_size
        self.nprobe = nprobe
        self.use_gpu = faiss.get_num_gpus() > 0 if use_gpu is None else use_gpu
        # The saved index, once loaded or saved.
        self.index = None

        self.d: int | None = None
        self.ntotal = 0
        # Each PDF's id, in id order.
        self.digest_ids: dict[str, int] = {}
        self._pending: list[tuple[np.ndarray, list[str], list]] = []
        # An empty trained index, which blocks are cloned from.
        self._trained: faiss.IndexIVFPQ | None = None
        # The trained index's quantizer, kept referenced for as long as the index.
        self._quantizer: faiss.Index | None = None
        self._assigner: faiss.Index | None = None
        self._block: faiss.Index | None = None
        self._block_paths: list[str] = []
        # The files rows are appended to, opened on the first write.
        self._files: dict | None = None

    def _path(self, name: str) -> str:
        return os.path.join(self.index_directory, name)

    # Building

    def build_index(self):
        return

    def add_batch(self, embeddings, pdf_names, pdf_pages):
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if self.d is None:
            self.d = vectors.shape[1]
        if self._trained is None and self.index is not None:
            # Adding to a saved index: encode with its training, in empty lists.
            trained = faiss.read_index(
                self._path(IVFPQ_INDEX_FILE), faiss.IO_FLAG_ONDISK_SAME_DIR
            )
            lists = faiss.ArrayInvertedLists(trained.nlist, trained.code_size)
            trained.replace_invlists(lists, True)
            lists.this.disown()
            trained.ntotal = 0
            self._use_trained(trained)
            # Rows and blocks written after the save are discarded, and the
            # PDFs that only they had.
            used = int(self.row_digests.max()) + 1 if self.ntotal else 0
            self.digest_ids = {d.decode(): i for i, d in enumerate(self.digests[:used])}
            for name, row_bytes in (
                (VECTORS_FILE, 2 * self.d),
                (ROW_DIGESTS_FILE, 4),
                (ROW_PAGES_FILE, 4),
            ):
                os.truncate(self._path(name), self.ntotal * row_bytes)
            for name in os.listdir(self.index_directory):
                if name.startswith("block_"):
                    os.remove(self._path(name))
        if self._trained is None:
            self._pending.append((vectors, pdf_names, pdf_pages))
            if sum(len(v) for v, _, _ in self._pending) >= self.train_size:
                self._train_and_add_pending()
            return
        self._encode(vectors, pdf_names, pdf_pages)

    def save_index(self):
        """Merge what was added with the lists already saved into a new on-disk
        index, write the row and PDF maps, and load it."""
        if self._trained is None:
            if self.index is not None:
                return
            if not self._pending:
                raise ValueError("No vectors were added")
            self._train_and_add_pending()
        if self._files is None:
            return  # Nothing was added since the index was saved.
        assert self._trained is not None
        self._flush_block()
        for file in self._files.values():
            file.flush()
            os.fsync(file.fileno())
            file.close()
        self._files = None

        digests = np.array(list(self.digest_ids), dtype=np.bytes_)
        np.save(self._path("digests.npy"), digests)
        # PDFs -> rows: all rows, by PDF id, and where each PDF's begin.
        row_digests = np.fromfile(self._path(ROW_DIGESTS_FILE), dtype=np.uint32)
        rows = np.argsort(row_digests, kind="stable").astype(np.uint32)
        offsets = np.zeros(len(digests) + 1, dtype=np.int64)
        np.cumsum(np.bincount(row_digests, minlength=len(digests)), out=offsets[1:])
        if self.index is not None:
            self.digest_rows.close()
        shutil.rmtree(self._path(DIGEST_ROWS_DB), ignore_errors=True)
        store = KVStore(self._path(DIGEST_ROWS_DB))
        # Appended in key order, in transactions of a million PDFs.
        order = np.argsort(digests)
        for start in range(0, len(order), 1_000_000):
            store.put_many(
                (
                    (bytes(digests[i]), rows[offsets[i] : offsets[i + 1]].tobytes())
                    for i in order[start : start + 1_000_000]
                ),
                append=True,
            )
        store.close()

        # The saved lists are read from disk and the blocks memory-mapped. (FAISS's
        # merge_ondisk() memory-maps every input, which crashes on saved lists.)
        saved = self._path(IVFPQ_INDEX_FILE)
        inputs = []
        if self.index is not None:
            inputs.append(faiss.read_index(saved, faiss.IO_FLAG_ONDISK_SAME_DIR))
        inputs += [
            faiss.read_index(path, faiss.IO_FLAG_MMAP) for path in self._block_paths
        ]
        lists = faiss.InvertedListsPtrVector()
        for index in inputs:
            lists.push_back(index.invlists)
        ivfdata = self._path(IVFPQ_DATA_FILE)
        merged = faiss.OnDiskInvertedLists(
            self._trained.nlist, self._trained.code_size, ivfdata + ".tmp"
        )
        index = faiss.clone_index(self._trained)
        index.ntotal = merged.merge_from_multiple(lists.data(), lists.size(), False)
        merged.filename = ivfdata
        index.replace_invlists(merged, True)
        merged.this.disown()
        faiss.write_index(index, saved + ".tmp")
        del inputs, index
        # The previous save is intact until these renames; load_index() finishes
        # them if they're interrupted.
        os.replace(ivfdata + ".tmp", ivfdata)
        os.replace(saved + ".tmp", saved)
        for path in self._block_paths:
            os.remove(path)
        self._block_paths = []
        logging.info(f"IVF-PQ index saved: {self.ntotal} rows, {len(digests)} PDFs")
        self.load_index()

    def _train_and_add_pending(self) -> None:
        assert self.d is not None
        d = self.d
        sample = np.concatenate([vectors for vectors, _, _ in self._pending])
        # Fewer lists than asked for if there isn't enough to train them on.
        max_lists = max(1, len(sample) // MIN_TRAIN_POINTS_PER_LIST)
        if self.nlist > max_lists:
            self.nlist = 1 << (max_lists.bit_length() - 1)
            logging.info(
                f"Only {len(sample)} training vectors: using {self.nlist} lists"
            )
        train = sample[: self.train_size]
        logging.info(
            f"Training {self.nlist} lists on {len(train)} vectors "
            f"({'GPU' if self.use_gpu else 'CPU'})"
        )
        kmeans = faiss.Kmeans(
            d, self.nlist, niter=20, seed=1234, gpu=self.use_gpu, verbose=False
        )
        kmeans.train(train)
        assert kmeans.centroids is not None
        quantizer = faiss.IndexFlatL2(d)
        quantizer.add(kmeans.centroids)
        index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, 8)
        # The quantizer is already trained and full, so this trains only the PQ
        # codebooks, on the residuals of a subsample.
        index.train(train[: min(len(train), 256 * 1024)])
        index.nprobe = self.nprobe
        self._use_trained(index)
        for vectors, digests, pages in self._pending:
            self._encode(vectors, digests, pages)
        self._pending = []

    def _use_trained(self, index: faiss.IndexIVFPQ) -> None:
        self._trained = index
        # Kept referenced for as long as the index; on the GPU when there is one.
        quantizer = faiss.downcast_index(index.quantizer)
        self._quantizer = quantizer
        self._assigner = (
            faiss.index_cpu_to_all_gpus(quantizer) if self.use_gpu else quantizer
        )

    def _encode(self, vectors: np.ndarray, digests: list[str], pages: list) -> None:
        from faiss.contrib.ivf_tools import add_preassigned

        assert self._trained is not None and self._assigner is not None
        if self._block is None:
            self._block = faiss.clone_index(self._trained)
        _, lists = self._assigner.search(vectors, 1)
        rows = np.arange(self.ntotal, self.ntotal + len(vectors), dtype=np.int64)
        add_preassigned(self._block, vectors, lists.ravel(), ids=rows)
        self.ntotal += len(vectors)
        if self.ntotal >= 2**32:
            raise ValueError("Rows are numbered with 32 bits")

        # A batch holds a few PDFs' pages: look up each distinct digest once, giving
        # new PDFs the next ids.
        unique, inverse = np.unique(np.asarray(digests), return_inverse=True)
        ids = np.array(
            [self.digest_ids.setdefault(str(d), len(self.digest_ids)) for d in unique],
            dtype=np.uint32,
        )
        digest_ids = ids[inverse.ravel()]
        if self._files is None:
            os.makedirs(self.index_directory, exist_ok=True)
            # Appended to a saved index's rows; kept open until save_index().
            mode = "ab" if self.index is not None else "wb"
            self._files = {
                name: open(self._path(name), mode)  # noqa: SIM115
                for name in (VECTORS_FILE, ROW_DIGESTS_FILE, ROW_PAGES_FILE)
            }
        self._files[VECTORS_FILE].write(vectors.astype(np.float16).tobytes())
        self._files[ROW_DIGESTS_FILE].write(digest_ids.tobytes())
        self._files[ROW_PAGES_FILE].write(np.asarray(pages, dtype=np.uint32).tobytes())
        if self._block.ntotal >= self.block_size:
            self._flush_block()

    def _flush_block(self) -> None:
        if self._block is None or self._block.ntotal == 0:
            return
        path = self._path(f"block_{len(self._block_paths):04d}.index")
        faiss.write_index(self._block, path)
        self._block_paths.append(path)
        self._block = None

    # Serving

    def load_index(self):
        # A save stopped between its renames (see save_index()) is finished, and
        # one stopped before them is discarded.
        saved, ivfdata = self._path(IVFPQ_INDEX_FILE), self._path(IVFPQ_DATA_FILE)
        if os.path.exists(saved + ".tmp") and not os.path.exists(ivfdata + ".tmp"):
            os.replace(saved + ".tmp", saved)
        for path in (saved + ".tmp", ivfdata + ".tmp"):
            if os.path.exists(path):
                os.remove(path)
        # ONDISK_SAME_DIR finds the lists next to ivfpq.index, wherever the
        # directory was moved; they're memory-mapped, not read in.
        self.index = faiss.read_index(
            saved, faiss.IO_FLAG_ONDISK_SAME_DIR | faiss.IO_FLAG_READ_ONLY
        )
        self.d, self.ntotal = d, ntotal = self.index.d, self.index.ntotal
        self.vectors = np.memmap(
            self._path(VECTORS_FILE), dtype=np.float16, mode="r", shape=(ntotal, d)
        )
        # Rows -> PDFs: each row's PDF (an index into digests) and page.
        self.row_digests = np.memmap(
            self._path(ROW_DIGESTS_FILE), dtype=np.uint32, mode="r", shape=(ntotal,)
        )
        self.row_pages = np.memmap(
            self._path(ROW_PAGES_FILE), dtype=np.uint32, mode="r", shape=(ntotal,)
        )
        self.digests = np.load(self._path("digests.npy"), mmap_mode="r")
        # Opened on first use, in each server worker.
        self.digest_rows = KVStore(self._path(DIGEST_ROWS_DB))

    def total_entries(self):
        # Including the vectors buffered to train on.
        return self.ntotal + sum(len(v) for v, _, _ in self._pending)

    def _rows_to_results(self, rows: np.ndarray) -> tuple[list[str], list[str]]:
        digests = self.digests[self.row_digests[rows]]
        return [d.decode() for d in digests], [str(p) for p in self.row_pages[rows]]

    def search(self, query_vector, k):
        query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        n_candidates = min(
            max(k * self.REFINE_FACTOR, self.MIN_CANDIDATES), self.total_entries()
        )
        _, ids = self.index.search(query, n_candidates)
        # Sorted, so the rerank reads the vector file in order.
        rows = np.sort(ids[0][ids[0] >= 0])
        if not len(rows):
            return np.empty(0, dtype=np.float32), [], []
        distances = ((self.vectors[rows].astype(np.float32) - query) ** 2).sum(axis=1)
        best = np.argsort(distances)[:k]
        digests, pages = self._rows_to_results(rows[best])
        return distances[best], digests, pages

    def get_vectors_for_digests(self, candidate_digests):
        values = self.digest_rows.get_many(str(d).encode() for d in candidate_digests)
        rows = [np.frombuffer(v, np.uint32) for v in values if v is not None]
        if not rows:
            return np.empty((0, 0), dtype=np.float32), [], []
        all_rows = np.sort(np.concatenate(rows))
        digests, pages = self._rows_to_results(all_rows)
        return self.vectors[all_rows].astype(np.float32), digests, pages
