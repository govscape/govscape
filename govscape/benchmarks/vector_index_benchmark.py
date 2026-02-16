"""Benchmark implementations of AbstractVectorIndex.

This script synthesizes random embedding batches alongside mock PDF metadata,
loads them into each available vector index backend, and reports ingestion and
query throughput metrics.

Example:
    poetry run python -m govscape.benchmarks.vector_index_benchmark \
        --embeddings 20000 --documents 5000 --dim 768 --queries 1000
"""

import argparse
import os
import random
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from govscape.indexing import AbstractVectorIndex, DiskANNIndex, FAISSIndex

IndexFactory = Callable[[Path], AbstractVectorIndex]


class BenchmarkDiskANNIndex(DiskANNIndex):
    """DiskANNIndex variant with a concrete add_batch implementation."""

    def __init__(self, embedding_directory: str, index_directory: str):
        super().__init__(embedding_directory, index_directory)
        self._entry_count = 0

    def add_batch(self, embeddings, pdf_names, pdf_pages):  # noqa: D401 - interface compliance
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        os.makedirs(self.embedding_directory, exist_ok=True)
        bin_path = os.path.join(self.embedding_directory, "embeddings.bin")
        with open(bin_path, "wb") as handle:
            np.array([vectors.shape[0]], dtype=np.int32).tofile(handle)
            np.array([vectors.shape[1]], dtype=np.int32).tofile(handle)
            vectors.tofile(handle)
        self._entry_count = vectors.shape[0]

    def total_entries(self):
        return self._entry_count


@dataclass
class BenchmarkResult:
    name: str
    embeddings: int
    dimension: int
    ingest_seconds: float
    ingest_embeddings_per_sec: float
    queries_per_sec: float
    avg_latency_ms: float


def generate_embeddings(num_embeddings: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_embeddings, dim)).astype(np.float32)


def generate_query_vectors(num_queries: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(num_queries, dim)).astype(np.float32)


def generate_metadata(
    num_documents: int, num_embeddings: int, seed: int
) -> tuple[list[str], list[int]]:
    if num_documents <= 0:
        raise ValueError("Number of documents must be positive")
    doc_names = [f"doc_{idx:05d}.pdf" for idx in range(num_documents)]
    rng = random.Random(seed)
    pdf_names: list[str] = []
    pdf_pages: list[int] = []
    for idx in range(num_embeddings):
        pdf_names.append(doc_names[idx % num_documents])
        pdf_pages.append(rng.randint(1, 40))
    return pdf_names, pdf_pages


def create_faiss_index(base_dir: Path) -> AbstractVectorIndex:
    base_dir.mkdir(parents=True, exist_ok=True)
    return FAISSIndex(base_dir.as_posix())


def create_diskann_index(base_dir: Path) -> AbstractVectorIndex:
    embedding_dir = base_dir / "embeddings"
    index_dir = base_dir / "index"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    return BenchmarkDiskANNIndex(embedding_dir.as_posix(), index_dir.as_posix())


INDEX_FACTORIES: dict[str, Callable[[Path], AbstractVectorIndex]] = {
    "faiss": create_faiss_index,
    "diskann": create_diskann_index,
}


def select_indexes(
    requested: Sequence[str],
) -> dict[str, Callable[[Path], AbstractVectorIndex]]:
    if not requested:
        return INDEX_FACTORIES
    selected: dict[str, Callable[[Path], AbstractVectorIndex]] = {}
    for name in requested:
        lowered = name.lower()
        if lowered not in INDEX_FACTORIES:
            raise ValueError(
                f"Unknown vector index '{name}'. Options: {sorted(INDEX_FACTORIES)}"
            )
        selected[lowered] = INDEX_FACTORIES[lowered]
    return selected


def benchmark_index(
    name: str,
    factory,
    embeddings: np.ndarray,
    pdf_names: Sequence[str],
    pdf_pages: Sequence[int],
    queries: np.ndarray,
    k: int,
    index_root: Path,
) -> BenchmarkResult:
    index_dir = index_root / name
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index = factory(index_dir)

    start_ingest = time.perf_counter()
    index.add_batch(embeddings, pdf_names, pdf_pages)
    index.build_index()
    ingest_seconds = time.perf_counter() - start_ingest

    # Ensure the index is ready for querying (DiskANN requires an explicit load step).
    if hasattr(index, "load_index"):
        try:
            index.load_index()
        except Exception as exc:
            raise RuntimeError(f"Failed to load index '{name}': {exc}") from exc

    latencies: list[float] = []
    for query in queries:
        q_start = time.perf_counter()
        index.search(query, k)
        latencies.append(time.perf_counter() - q_start)

    total_query_time = sum(latencies) or 1e-9
    queries_per_sec = len(queries) / total_query_time
    avg_latency_ms = (statistics.mean(latencies) * 1000) if latencies else 0.0

    ingest_embeddings_per_sec = (
        embeddings.shape[0] / ingest_seconds if ingest_seconds else float("inf")
    )
    return BenchmarkResult(
        name=name,
        embeddings=embeddings.shape[0],
        dimension=embeddings.shape[1],
        ingest_seconds=ingest_seconds,
        ingest_embeddings_per_sec=ingest_embeddings_per_sec,
        queries_per_sec=queries_per_sec,
        avg_latency_ms=avg_latency_ms,
    )


def format_results(results: Iterable[BenchmarkResult]) -> str:
    header = (
        f"{'Index':<12} {'Embeddings':>12} {'Dim':>6} {'Ingest (s)':>12} "
        f"{'Emb/s':>12} {'Queries/s':>12} {'Avg Latency (ms)':>18}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(
        f"{res.name:<12} {res.embeddings:>12} {res.dimension:>6} "
        f"{res.ingest_seconds:>12.4f} {res.ingest_embeddings_per_sec:>12.2f} "
        f"{res.queries_per_sec:>12.2f} {res.avg_latency_ms:>18.2f}"
        for res in results
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vector index implementations"
    )
    parser.add_argument("--dim", type=int, default=768, help="Embedding dimensionality")
    parser.add_argument(
        "--embeddings", type=int, default=50000, help="Number of embeddings to index"
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=5000,
        help="Number of unique document identifiers",
    )
    parser.add_argument(
        "--queries", type=int, default=200, help="Number of random query vectors"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of neighbors to retrieve per query"
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("./.vector_index_benchmark"),
        help="Directory where temporary index artifacts will be stored",
    )
    parser.add_argument(
        "--indexes",
        nargs="*",
        default=None,
        help=f"Subset of indexes to run ({', '.join(sorted(INDEX_FACTORIES))})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args(argv)


def try_benchmark(
    name: str,
    factory: IndexFactory,
    embeddings: np.ndarray,
    pdf_names: Sequence[str],
    pdf_pages: Sequence[int],
    queries: np.ndarray,
    k: int,
    index_root: Path,
) -> BenchmarkResult | None:
    try:
        return benchmark_index(
            name,
            factory,
            embeddings,
            pdf_names,
            pdf_pages,
            queries,
            k,
            index_root,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Skipping {name} due to error: {exc}", file=sys.stderr)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.embeddings <= 0:
        raise ValueError("Number of embeddings must be positive")
    if args.dim <= 0:
        raise ValueError("Embedding dimension must be positive")
    if args.documents <= 0:
        raise ValueError("Number of documents must be positive")

    rng = random.Random(args.seed)
    embedding_seed = rng.randrange(1 << 63)
    query_seed = rng.randrange(1 << 63)
    metadata_seed = rng.randrange(1 << 63)

    embeddings = generate_embeddings(args.embeddings, args.dim, embedding_seed)
    queries = generate_query_vectors(args.queries, args.dim, query_seed)
    pdf_names, pdf_pages = generate_metadata(
        args.documents, args.embeddings, metadata_seed
    )

    selected = select_indexes(args.indexes or [])
    index_root = args.index_root.resolve()
    index_root.mkdir(parents=True, exist_ok=True)

    k = max(1, min(args.k, args.embeddings))
    results: list[BenchmarkResult] = []
    for name, factory in selected.items():
        result = try_benchmark(
            name,
            factory,
            embeddings,
            pdf_names,
            pdf_pages,
            queries,
            k,
            index_root,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("No vector index benchmarks completed successfully.", file=sys.stderr)
        return 1

    print(format_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
