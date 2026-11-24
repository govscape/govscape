"""A disk-based IVF-PQ vector index for archive-scale serving.

The index is built once by IVFPQIndexBuilder and served read-only by
IVFPQDiskIndex, with almost everything memory-mapped from local disk:

- ivfpq.index, ivfpq.ivfdata: a FAISS IndexIVFPQ whose inverted lists (each
  vector's 64-byte PQ code and row id) live in ivfpq.ivfdata and are read from
  disk at search time; only the coarse centroids are held in memory.
- vectors.f16: every vector in float16, by row, to rerank PQ candidates exactly
  and to return all of a PDF's page vectors for filtered (prefilter) searches.
- row_digests.u32, row_pages.u32: each row's PDF (an index into digests.npy)
  and page number.
- digests.npy (PDF digests by index), digests_sorted.npy and
  digests_sorted_ids.npy (to look a digest up), digest_rows.npy and
  digest_row_offsets.npy (each PDF's rows): the PDF -> rows index.
- meta.json: dimensions, sizes and search settings.

Building uses the GPU when FAISS has one (faiss-gpu): to train the coarse
quantizer with k-means and to assign every vector to its list, which dominate
the build at this scale. PQ encoding and the list files are done on the CPU.
"""

import itertools
import json
import logging
import os
from typing import cast

import numpy as np

from .vector import AbstractVectorIndex, faiss

INDEX_FILE = "ivfpq.index"
IVFDATA_FILE = "ivfpq.ivfdata"
META_FILE = "meta.json"
VECTORS_FILE = "vectors.f16"
ROW_DIGESTS_FILE = "row_digests.u32"
ROW_PAGES_FILE = "row_pages.u32"
# A build's checkpoint, removed when the build finishes.
STATE_FILE = "build_state.json"
TRAINED_FILE = "build_trained.index"
DIGEST_LIST_FILE = "build_digests.txt"

# FAISS recommends at least this many training points per coarse centroid.
MIN_TRAIN_POINTS_PER_LIST = 39


class IVFPQIndexBuilder:
    """Builds an IVFPQDiskIndex in index_directory from (vectors, digests, pages)
    batches passed to add(), in a single pass over the data.

    The first train_size vectors are buffered to train the index, so they
    should be a fair sample of all of them (digests are hashes, so any shard
    is). Encoded vectors are written in blocks of block_size, which finish()
    merges into one on-disk index.

    Once checkpoint_due (every checkpoint_every vectors), the caller can
    save_checkpoint() with its progress through its input; resume() then
    restores the builder and that progress after a restart.
    """

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
        os.makedirs(index_directory, exist_ok=True)

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

    @classmethod
    def resume(
        cls, index_directory: str, use_gpu: bool | None = None
    ) -> tuple["IVFPQIndexBuilder", dict] | None:
        """The builder last checkpointed in index_directory, with the progress
        saved with it, or None if there's no checkpoint."""
        path = os.path.join(index_directory, STATE_FILE)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            state = json.load(f)
        builder = cls(index_directory, use_gpu=use_gpu, **state["config"])
        # Anything written after the checkpoint is discarded: the files are cut
        # back to their checkpointed lengths and later blocks deleted.
        builder.d = state["d"]
        builder.ntotal = state["ntotal"]
        builder._next_checkpoint = builder.ntotal + builder.checkpoint_every
        builder._files = {}
        for name, size in state["file_bytes"].items():
            # Kept open for the rest of the build; finish() closes them.
            file = open(builder._path(name), "r+b")  # noqa: SIM115
            file.truncate(size)
            file.seek(size)
            builder._files[name] = file
        builder._block_paths = [builder._path(name) for name in state["blocks"]]
        for name in os.listdir(index_directory):
            if name.startswith("block_") and name not in state["blocks"]:
                os.remove(builder._path(name))
        with open(builder._path(DIGEST_LIST_FILE)) as f:
            digests = [line.rstrip("\n") for line in f][: state["n_digests"]]
        with open(builder._path(DIGEST_LIST_FILE), "w") as f:
            f.writelines(d + "\n" for d in digests)
        builder.digest_ids = {digest: i for i, digest in enumerate(digests)}
        builder._digests_saved = len(digests)
        # read_index returns the concrete type; this tells mypy so (wrapping it
        # in downcast_index would leave the index owned by a discarded object).
        builder._use_trained(
            cast(faiss.IndexIVFPQ, faiss.read_index(builder._path(TRAINED_FILE)))
        )
        return builder, state["progress"]

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

    def _path(self, name: str) -> str:
        return os.path.join(self.index_directory, name)

    def add(self, vectors: np.ndarray, digests: list[str], pages: list) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.d is None:
            self.d = vectors.shape[1]
        if self._trained is None:
            self._pending.append((vectors, digests, pages))
            self._pending_count += len(vectors)
            if self._pending_count >= self.train_size:
                self._train_and_add_pending()
            return
        self._encode(vectors, digests, pages)

    def finish(self) -> None:
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
        merge_ondisk(index, self._block_paths, self._path(IVFDATA_FILE))
        faiss.write_index(index, self._path(INDEX_FILE))
        for path in self._block_paths:
            os.remove(path)

        # PDFs -> rows (see IVFPQDiskIndex.load_index).
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
        with open(self._path(META_FILE), "w") as f:
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
            # Kept open for the whole build; finish() closes them.
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


class IVFPQDiskIndex(AbstractVectorIndex):
    """Serves an index built by IVFPQIndexBuilder from index_directory, reading
    its lists and vectors from disk (see the module docstring)."""

    # PQ candidates per result, reranked with the exact (float16) vectors. On the
    # 156k-page pilot (2,048 lists), 1,000 candidates and nprobe 128 found 98% of
    # the exact top 10, against 86% with 100 candidates.
    REFINE_FACTOR = 50
    MIN_CANDIDATES = 1000

    def __init__(self, index_directory, nprobe: int | None = None):
        self.index_directory = index_directory
        self.nprobe = nprobe
        self.index = None

    def _path(self, name: str) -> str:
        return os.path.join(self.index_directory, name)

    def load_index(self):
        with open(self._path(META_FILE)) as f:
            self.meta = json.load(f)
        d, ntotal = self.meta["d"], self.meta["ntotal"]
        # ONDISK_SAME_DIR finds ivfpq.ivfdata next to ivfpq.index, wherever the
        # directory was moved; the lists are memory-mapped, not read in.
        self.index = faiss.read_index(
            self._path(INDEX_FILE),
            faiss.IO_FLAG_ONDISK_SAME_DIR | faiss.IO_FLAG_READ_ONLY,
        )
        self.index.nprobe = self.nprobe or self.meta["nprobe"]
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
