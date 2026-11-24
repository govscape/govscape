"""Build a disk-based IVF-PQ index (govscape/indexing/ivfpq.py) of a serving
directory's text or page-image embeddings, and upload it to the serving
directory's index directory (index/ or index_img_pg/).

The embeddings are read once, straight from the pipeline's chunks (tar.gz
archives of per-page .npy files): chunks are downloaded in threads and parsed in
processes, a window at a time, and their vectors go to the builder in order.
The build uses the GPU when FAISS has one. Every --checkpoint_every vectors it
checkpoints in the local index directory, and a restarted build resumes from
there (a spot builder keeps its root volume when it's interrupted); --restart
ignores the checkpoint.
"""

import hashlib
import io
import logging
import os
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np

from govscape.config import DataModel
from govscape.data_loader import build_data_loader
from govscape.indexing.ivfpq import IVFPQIndexBuilder
from govscape.utils import base_argument_parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Chunks downloaded and parsed together before their vectors go to the builder.
CHUNKS_PER_WINDOW = 64


def read_chunk(path: str) -> tuple[np.ndarray, list[str], list[int]]:
    """The vectors in an embedding chunk, with each one's digest and page, from
    members named <digest>/<digest>_<page>.npy."""
    vectors, digests, pages = [], [], []
    with tarfile.open(path) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".npy"):
                continue
            file = tar.extractfile(member)
            assert file is not None
            vectors.append(np.load(io.BytesIO(file.read())).ravel())
            name = os.path.basename(member.name)[: -len(".npy")]
            digest, _, page = name.rpartition("_")
            digests.append(digest)
            pages.append(int(page))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), [], []
    return np.stack(vectors).astype(np.float32), digests, pages


def chunks_digest(chunks: list[str]) -> str:
    return hashlib.sha1("\n".join(chunks).encode()).hexdigest()


def main():
    parser = base_argument_parser(description="Build a disk-based IVF-PQ index")
    parser.add_argument("--embedding_type", choices=["txt", "img_pg"], required=True)
    parser.add_argument("--nlist", type=int, default=131072, help="Inverted lists")
    parser.add_argument(
        "--m", type=int, default=64, help="PQ code size in bytes (subquantizers)"
    )
    parser.add_argument(
        "--train_size", type=int, default=None, help="Default: 50 per list"
    )
    parser.add_argument("--block_size", type=int, default=20_000_000)
    parser.add_argument(
        "--nprobe", type=int, default=128, help="Lists searched per query (default)"
    )
    parser.add_argument(
        "--max_chunks", type=int, default=None, help="Only index the first chunks"
    )
    parser.add_argument("--checkpoint_every", type=int, default=100_000_000)
    parser.add_argument(
        "--restart", action="store_true", help="Ignore an existing checkpoint"
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    local_dm = DataModel(os.path.join(project_root, "data", "prod"))
    remote_dm = DataModel(args.remote_data_dir)
    if args.embedding_type == "txt":
        remote_embeddings = remote_dm.embedding_directory
        local_index, remote_index = local_dm.index_directory, remote_dm.index_directory
    else:
        remote_embeddings = remote_dm.embedding_img_pg_directory
        local_index = local_dm.index_img_pg_directory
        remote_index = remote_dm.index_img_pg_directory

    data_loader = build_data_loader(
        args.backend,
        args.bucket_name,
        local_base_dir=args.local_base_dir,
        profile_name=args.profile,
        endpoint_url=args.endpoint_url,
    )
    chunks, token = [], None
    while True:
        listing = data_loader.list_objects(
            remote_embeddings, max_keys=1000, continuation_token=token
        )
        chunks += [key for key in listing.keys if key.endswith(".tar.gz")]
        if not listing.is_truncated:
            break
        token = listing.continuation_token
    chunks = sorted(chunks)[: args.max_chunks]
    logging.info(f"{len(chunks)} embedding chunks in {remote_embeddings}")

    resumed = None if args.restart else IVFPQIndexBuilder.resume(local_index)
    if resumed is not None:
        builder, progress = resumed
        chunks_done = progress["chunks_done"]
        # The chunks are listed afresh, so check those already indexed still
        # come first (new chunks may have been added after them).
        if chunks_digest(chunks[:chunks_done]) != progress["chunks_digest"]:
            raise RuntimeError(
                "The embedding chunks changed since the checkpoint; rerun with "
                "--restart"
            )
        logging.info(f"Resuming after {chunks_done} chunks, {builder.ntotal} vectors")
    else:
        shutil.rmtree(local_index, ignore_errors=True)
        builder = IVFPQIndexBuilder(
            local_index,
            nlist=args.nlist,
            m=args.m,
            train_size=args.train_size,
            block_size=args.block_size,
            nprobe=args.nprobe,
            checkpoint_every=args.checkpoint_every,
        )
        chunks_done = 0
    start = time.time()
    with (
        tempfile.TemporaryDirectory() as download_dir,
        ThreadPoolExecutor(max_workers=16) as downloads,
        ProcessPoolExecutor(mp_context=get_context("spawn")) as parsers,
    ):

        def download(key: str) -> str:
            path = os.path.join(download_dir, os.path.basename(key))
            data_loader.download_file(key, path)
            return path

        for first in range(chunks_done, len(chunks), CHUNKS_PER_WINDOW):
            paths = list(
                downloads.map(download, chunks[first : first + CHUNKS_PER_WINDOW])
            )
            for vectors, digests, pages in parsers.map(read_chunk, paths):
                if len(vectors):
                    builder.add(vectors, digests, pages)
            for path in paths:
                os.remove(path)
            done = first + len(paths)
            if builder.checkpoint_due:
                builder.save_checkpoint(
                    {"chunks_done": done, "chunks_digest": chunks_digest(chunks[:done])}
                )
            logging.info(
                f"{done}/{len(chunks)} chunks, {builder.ntotal} vectors "
                f"encoded, {time.time() - start:.0f} s"
            )
    builder.finish()
    logging.info(f"Built in {time.time() - start:.0f} s; uploading to {remote_index}")
    data_loader.upload_directory(local_index, remote_index)
    logging.info(f"Done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
