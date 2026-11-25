import contextlib
import itertools
import json
import logging
import os
import pickle as pkl
import sys
from abc import ABC, abstractmethod
from typing import cast

import numpy as np

import pyarrow as pa

from .base import AbstractIndex
from .forward import build_forward_index


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
    """What the server and the hybrid indexes need from a vector index: load it,
    search it, and look up every page vector of given PDFs (for prefiltering)."""

    @abstractmethod
    def load_index(self):
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


class BuildableVectorIndex(AbstractVectorIndex):
    """A vector index built with build_index(), then add_batch() for every batch
    of page vectors, then save_index()."""

    @abstractmethod
    def build_index(self):
        pass

    @abstractmethod
    def add_batch(self, embeddings, pdf_names, pdf_pages):
        pass

    @abstractmethod
    def save_index(self):
        pass


class FAISSIndex(BuildableVectorIndex):
    def __init__(self, index_directory, index_type="IVFPQ", forward_index_type="LMDB"):
        self.index_directory = index_directory
        self.forward_index = build_forward_index(forward_index_type, index_directory)
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
        self.forward_index.add_batch(embeddings, pdf_names, pdf_pages)

    def build_index(self):
        return

    def save_index(self):
        os.makedirs(self.index_directory, exist_ok=True)
        index_path = os.path.join(self.index_directory, "faiss_index.pkl")
        with open(index_path, "wb") as handle:
            pkl.dump(self, handle)
        self.forward_index.save_index()
        print(f"Index saved to {self.index_directory}/faiss_index.pkl")
        return

    def load_index(self):
        self.forward_index.load_index()
        index_path = os.path.join(self.index_directory, "faiss_index.pkl")
        if not os.path.exists(index_path):
            return
        with open(index_path, "rb") as handle:
            index = pkl.load(handle)
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
        return self.forward_index.get_vectors_for_digests(candidate_digests)


class LanceDBVectorIndex(BuildableVectorIndex):
    def __init__(
        self, index_directory, table_name="vector_index", forward_index_type="LMDB"
    ):
        self.index_directory = index_directory
        self.forward_index = build_forward_index(forward_index_type, index_directory)
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
        self.forward_index.add_batch(embeddings, pdf_names, pdf_pages)

    def save_index(self):
        self.forward_index.save_index()
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
        self.forward_index.load_index()
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
        return self.forward_index.get_vectors_for_digests(candidate_digests)


IVFPQ_INDEX_FILE = "ivfpq.index"
IVFPQ_DATA_FILE = "ivfpq.ivfdata"
IVFPQ_META_FILE = "meta.json"
VECTORS_FILE = "vectors.f16"
ROW_DIGESTS_FILE = "row_digests.u32"
ROW_PAGES_FILE = "row_pages.u32"
# A build's checkpoint, removed when the build finishes.
STATE_FILE = "build_state.json"
TRAINED_FILE = "build_trained.index"
DIGEST_LIST_FILE = "build_digests.txt"
# FAISS recommends at least this many training points per coarse centroid.
MIN_TRAIN_POINTS_PER_LIST = 39


class IVFPQDiskIndex(BuildableVectorIndex):
    """A disk-based IVF-PQ index for archive-scale serving, built in a single pass
    over the data and served with almost everything memory-mapped from disk:

    - ivfpq.index, ivfpq.ivfdata: a FAISS IndexIVFPQ whose inverted lists (each
      vector's PQ code and row id) live in ivfpq.ivfdata and are read from disk
      at search time; only the coarse centroids are held in memory.
    - vectors.f16: every vector in float16, by row, to rerank PQ candidates
      exactly and to return all of a PDF's page vectors for prefiltering.
    - row_digests.u32, row_pages.u32: each row's PDF (an index into digests.npy)
      and page number.
    - digests.npy (PDF digests by id), digests_sorted.npy and
      digests_sorted_ids.npy (to look a digest up), digest_rows.npy and
      digest_row_offsets.npy (each PDF's rows): the PDF -> rows index.
    - meta.json: dimensions, sizes and search settings.

    Building: the first train_size vectors passed to add_batch() are buffered to
    train the index, so they should be a fair sample (digests are hashes, so any
    shard is). FAISS trains the coarse quantizer and assigns vectors to lists on
    the GPU when it has one; PQ encoding is done on the CPU. Encoded vectors are
    written in blocks of block_size, which save_index() merges into one on-disk
    index. Once checkpoint_due (every checkpoint_every vectors), the caller can
    save_checkpoint() with its progress through its input; resume() then restores
    the index being built and that progress after a restart.
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
        checkpoint_every: int = 100_000_000,
    ):
        self.index_directory = index_directory
        self.nlist = nlist
        self.m = m
        self.train_size = train_size or 50 * nlist
        self.block_size = block_size
        self.nprobe = nprobe
        self.use_gpu = faiss.get_num_gpus() > 0 if use_gpu is None else use_gpu
        self.checkpoint_every = checkpoint_every
        self.index = None

        self.d: int | None = None
        self.ntotal = 0
        self._next_checkpoint = checkpoint_every
        # Each PDF's id, in id order; the first _digests_saved are checkpointed.
        self.digest_ids: dict[str, int] = {}
        self._digests_saved = 0
        self._pending: list[tuple[np.ndarray, list[str], list]] = []
        self._pending_count = 0
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

    @classmethod
    def resume(
        cls, index_directory: str, use_gpu: bool | None = None
    ) -> tuple["IVFPQDiskIndex", dict] | None:
        """The index last checkpointed in index_directory, with the progress saved
        with it, or None if there's no checkpoint."""
        path = os.path.join(index_directory, STATE_FILE)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            state = json.load(f)
        index = cls(index_directory, use_gpu=use_gpu, **state["config"])
        # Anything written after the checkpoint is discarded: the files are cut
        # back to their checkpointed lengths and later blocks deleted.
        index.d = state["d"]
        index.ntotal = state["ntotal"]
        index._next_checkpoint = index.ntotal + index.checkpoint_every
        index._files = {}
        for name, size in state["file_bytes"].items():
            # Kept open for the rest of the build; save_index() closes them.
            file = open(index._path(name), "r+b")  # noqa: SIM115
            file.truncate(size)
            file.seek(size)
            index._files[name] = file
        index._block_paths = [index._path(name) for name in state["blocks"]]
        for name in os.listdir(index_directory):
            if name.startswith("block_") and name not in state["blocks"]:
                os.remove(index._path(name))
        with open(index._path(DIGEST_LIST_FILE)) as f:
            digests = [line.rstrip("\n") for line in f][: state["n_digests"]]
        with open(index._path(DIGEST_LIST_FILE), "w") as f:
            f.writelines(d + "\n" for d in digests)
        index.digest_ids = {digest: i for i, digest in enumerate(digests)}
        index._digests_saved = len(digests)
        # read_index returns the concrete type; this tells mypy so (wrapping it
        # in downcast_index would leave the index owned by a discarded object).
        index._use_trained(
            cast(faiss.IndexIVFPQ, faiss.read_index(index._path(TRAINED_FILE)))
        )
        return index, state["progress"]

    @property
    def checkpoint_due(self) -> bool:
        return self._trained is not None and self.ntotal >= self._next_checkpoint

    def save_checkpoint(self, progress: dict) -> None:
        """Make everything added so far survive a restart (see resume()), with
        the caller's progress through its input."""
        assert self._trained is not None and self._files is not None
        self._flush_block()
        for file in self._files.values():
            file.flush()
            os.fsync(file.fileno())
        if not os.path.exists(self._path(TRAINED_FILE)):
            faiss.write_index(self._trained, self._path(TRAINED_FILE))
        with open(self._path(DIGEST_LIST_FILE), "a") as f:
            new = itertools.islice(self.digest_ids, self._digests_saved, None)
            f.writelines(d + "\n" for d in new)
            f.flush()
            os.fsync(f.fileno())
        self._digests_saved = len(self.digest_ids)
        config = (
            "nlist",
            "m",
            "train_size",
            "block_size",
            "nprobe",
            "checkpoint_every",
        )
        state = {
            "config": {name: getattr(self, name) for name in config},
            "d": self.d,
            "ntotal": self.ntotal,
            "n_digests": len(self.digest_ids),
            "blocks": [os.path.basename(p) for p in self._block_paths],
            "file_bytes": {name: f.tell() for name, f in self._files.items()},
            "progress": progress,
        }
        # Written last and atomically: the checkpoint is whatever it describes.
        with open(self._path(STATE_FILE) + ".tmp", "w") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self._path(STATE_FILE) + ".tmp", self._path(STATE_FILE))
        self._next_checkpoint = self.ntotal + self.checkpoint_every
        logging.info(f"Checkpointed {self.ntotal} vectors: {progress}")

    def build_index(self):
        return

    def add_batch(self, embeddings, pdf_names, pdf_pages):
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if self.d is None:
            self.d = vectors.shape[1]
        if self._trained is None:
            self._pending.append((vectors, pdf_names, pdf_pages))
            self._pending_count += len(vectors)
            if self._pending_count >= self.train_size:
                self._train_and_add_pending()
            return
        self._encode(vectors, pdf_names, pdf_pages)

    def save_index(self):
        """Encode what's left, merge the blocks into the on-disk index and write
        the row and PDF maps."""
        if self._trained is None:
            if not self._pending:
                raise ValueError("No vectors were added")
            self._train_and_add_pending()
        assert self._trained is not None and self._files is not None
        self._flush_block()
        for file in self._files.values():
            file.close()

        from faiss.contrib.ondisk import merge_ondisk

        index = self._trained
        merge_ondisk(index, self._block_paths, self._path(IVFPQ_DATA_FILE))
        faiss.write_index(index, self._path(IVFPQ_INDEX_FILE))
        for path in self._block_paths:
            os.remove(path)

        # PDFs -> rows (see load_index).
        digests = np.array(list(self.digest_ids), dtype=np.bytes_)
        np.save(self._path("digests.npy"), digests)
        order = np.argsort(digests).astype(np.uint32)
        np.save(self._path("digests_sorted.npy"), digests[order])
        np.save(self._path("digests_sorted_ids.npy"), order)
        row_digests = np.fromfile(self._path(ROW_DIGESTS_FILE), dtype=np.uint32)
        row_dtype = np.uint32 if self.ntotal < 2**32 else np.uint64
        digest_rows = np.argsort(row_digests, kind="stable").astype(row_dtype)
        offsets = np.zeros(len(digests) + 1, dtype=np.uint64)
        np.cumsum(np.bincount(row_digests, minlength=len(digests)), out=offsets[1:])
        np.save(self._path("digest_rows.npy"), digest_rows)
        np.save(self._path("digest_row_offsets.npy"), offsets)

        meta = {
            "d": self.d,
            "nlist": self.nlist,
            "m": self.m,
            "ntotal": self.ntotal,
            "n_digests": len(self.digest_ids),
            "nprobe": self.nprobe,
        }
        with open(self._path(IVFPQ_META_FILE), "w") as f:
            json.dump(meta, f, indent=2)
        for name in (STATE_FILE, TRAINED_FILE, DIGEST_LIST_FILE):
            if os.path.exists(self._path(name)):
                os.remove(self._path(name))
        logging.info(f"IVF-PQ index written: {meta}")

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
            # Kept open for the whole build; save_index() closes them.
            self._files = {
                name: open(self._path(name), "wb")  # noqa: SIM115
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
        with open(self._path(IVFPQ_META_FILE)) as f:
            self.meta = json.load(f)
        d, ntotal = self.meta["d"], self.meta["ntotal"]
        # ONDISK_SAME_DIR finds ivfpq.ivfdata next to ivfpq.index, wherever the
        # directory was moved; the lists are memory-mapped, not read in.
        self.index = faiss.read_index(
            self._path(IVFPQ_INDEX_FILE),
            faiss.IO_FLAG_ONDISK_SAME_DIR | faiss.IO_FLAG_READ_ONLY,
        )
        self.index.nprobe = self.meta["nprobe"]
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
        # PDFs -> rows: a PDF's id by its digest (binary search over the sorted
        # digests), then its rows, digest_rows[offsets[id]:offsets[id + 1]].
        self.digests = self._load("digests")
        self.digests_sorted = self._load("digests_sorted")
        self.digests_sorted_ids = self._load("digests_sorted_ids")
        self.digest_rows = self._load("digest_rows")
        self.digest_row_offsets = self._load("digest_row_offsets")

    def _load(self, name: str) -> np.ndarray:
        return np.load(self._path(f"{name}.npy"), mmap_mode="r")

    def total_entries(self):
        return self.meta["ntotal"]

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
        # Each PDF's id by binary search over the sorted digests, then its rows.
        rows = []
        for digest in candidate_digests:
            key = str(digest).encode()
            position = np.searchsorted(self.digests_sorted, key)
            if (
                position < len(self.digests_sorted)
                and self.digests_sorted[position] == key
            ):
                digest_id = self.digests_sorted_ids[position]
                start, end = self.digest_row_offsets[digest_id : digest_id + 2]
                rows.append(np.asarray(self.digest_rows[start:end]))
        if not rows:
            return np.empty((0, 0), dtype=np.float32), [], []
        all_rows = np.sort(np.concatenate(rows))
        digests, pages = self._rows_to_results(all_rows)
        return self.vectors[all_rows].astype(np.float32), digests, pages
