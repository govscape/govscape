"""Benchmark implementations of AbstractVectorIndex.

This script synthesizes random embedding batches alongside mock PDF metadata,
loads them into each available vector index backend, and reports ingestion and
query throughput metrics.

Example:
    poetry run python -m govscape.benchmarks.vector_index_benchmark \
        --embeddings 20000 --documents 5000 --dim 768 --queries 1000
"""

import argparse
import random
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from govscape.data_loader import RemoteDirectoryIterator, build_data_loader
from govscape.indexing import AbstractVectorIndex, FAISSIndex, LanceDBVectorIndex

IndexFactory = Callable[[Path], AbstractVectorIndex]


@dataclass
class BenchmarkResult:
    name: str
    embeddings: int
    dimension: int
    ingest_seconds: float
    ingest_embeddings_per_sec: float
    index_size_mb: float
    queries_per_sec: float
    avg_latency_ms: float
    recall_at_k: float | None


def compute_directory_size_mb(directory: Path) -> float:
    """Return total size of files under directory in MB."""
    if not directory.exists():
        return 0.0
    total_bytes = sum(
        file_path.stat().st_size
        for file_path in directory.rglob("*")
        if file_path.is_file()
    )
    return total_bytes / (1024.0 * 1024.0)


def compute_exact_topk_threshold(
    embeddings: np.ndarray,
    queries: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute exact nearest-neighbor ids for each query under L2 distance."""
    if embeddings.ndim != 2 or queries.ndim != 2:
        raise ValueError("Embeddings and queries must be 2D arrays")
    if embeddings.shape[1] != queries.shape[1]:
        raise ValueError(
            "Embedding dimension mismatch between embeddings and queries: "
            f"{embeddings.shape[1]} vs {queries.shape[1]}"
        )

    k = max(1, min(k, embeddings.shape[0]))
    index = FAISSIndex("/tmp/benchmark_temp_index", index_type="Flat")
    index.add_batch(
        embeddings, ["temp"] * embeddings.shape[0], [1] * embeddings.shape[0]
    )
    index.build_index()
    D, _ = index.faiss_index.search(queries, k)
    return D[:, -1]


def extract_result_pairs(
    search_result: tuple,
    k: int,
) -> list[tuple[str, int]]:
    """Extract (pdf_name, page) pairs from the AbstractVectorIndex search output."""
    if len(search_result) < 3:
        raise RuntimeError(
            "Cannot compute recall@k: AbstractVectorIndex.search() output is "
            "ambiguous for this backend. Expected a tuple of "
            "(distances, pdf_names, pages)."
        )

    _, names, pages = search_result
    pairs = [(str(name), int(page)) for name, page in zip(names, pages, strict=False)]
    return pairs[:k]


def extract_result_distances(search_result: tuple, k: int) -> np.ndarray:
    """Extract distances from the AbstractVectorIndex search output."""
    if len(search_result) < 1:
        raise RuntimeError(
            "Cannot compute recall@k: AbstractVectorIndex.search() output is "
            "ambiguous for this backend. Expected distances in the first "
            "tuple element."
        )

    distances = np.asarray(search_result[0], dtype=np.float32).reshape(-1)
    return distances[:k]


def compute_recall_at_k(
    predicted_distances: Sequence[np.ndarray],
    exact_kth_distances: np.ndarray,
    k: int,
    atol: float = 1e-6,
) -> float:
    """Compute mean distance-based recall@k across queries.

    A hit is counted when a predicted neighbor distance is <= the exact k-th
    distance for that query (with tolerance). This is tie-aware and avoids
    penalizing equivalent neighbors with the same distance.
    """
    if not predicted_distances:
        return 0.0
    total = min(len(predicted_distances), exact_kth_distances.shape[0])
    if total == 0:
        return 0.0

    recalls: list[float] = []
    for idx in range(total):
        pred = predicted_distances[idx][:k]
        threshold = float(exact_kth_distances[idx]) + atol
        hits = int(np.sum(pred <= threshold))
        recalls.append(hits / k)
    return float(statistics.mean(recalls))


def generate_embeddings(num_embeddings: int, dim: int, seed: int) -> np.ndarray:
    data_loader = build_data_loader("s3", bucket_name="bcgl-public-bucket")
    iterator = RemoteDirectoryIterator(
        data_loader=data_loader,
        prefix="test-serving/embeddings_compressed/",
        remote_checkpoint_path="test-serving/Checkpoints/benchmark_checkpoint.json",
        local_checkpoint_path="/tmp/Checkpoints/vector_benchmark_checkpoint.json",
        local_dir="/tmp/data/vector_benchmark_embeddings",
    )

    embeddings: list[np.ndarray] = []
    while len(embeddings) < num_embeddings:
        batch = iterator.download_batch(
            max_keys=min(100000, int((num_embeddings - len(embeddings)) / 1000)),
            num_workers=64,
        )
        if len(batch) == 0:
            break
        embeddings.extend([np.load(path) for path in batch])
        print("Downloaded batch of embeddings, total so far:", len(embeddings))
    if len(embeddings) < num_embeddings:
        raise ValueError(f"Expected {num_embeddings} embeddings, got {len(embeddings)}")
    return np.array(embeddings[:num_embeddings], dtype=np.float32)


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


def create_ivfpq_faiss_index(base_dir: Path) -> AbstractVectorIndex:
    base_dir.mkdir(parents=True, exist_ok=True)
    return FAISSIndex(base_dir.as_posix(), index_type="IVFPQ")


def create_ivf_faiss_index(base_dir: Path) -> AbstractVectorIndex:
    base_dir.mkdir(parents=True, exist_ok=True)
    return FAISSIndex(base_dir.as_posix(), index_type="IVF")


def create_hnsw_faiss_index(base_dir: Path) -> AbstractVectorIndex:
    base_dir.mkdir(parents=True, exist_ok=True)
    return FAISSIndex(base_dir.as_posix(), index_type="HNSW")


def create_flat_faiss_index(base_dir: Path) -> AbstractVectorIndex:
    base_dir.mkdir(parents=True, exist_ok=True)
    return FAISSIndex(base_dir.as_posix(), index_type="Flat")


def create_lancedb_index(base_dir: Path) -> AbstractVectorIndex:
    base_dir.mkdir(parents=True, exist_ok=True)
    return LanceDBVectorIndex(base_dir.as_posix())


INDEX_FACTORIES: dict[str, Callable[[Path], AbstractVectorIndex]] = {
    "faiss_IVFPQ": create_ivfpq_faiss_index,
    "faiss_IVF": create_ivf_faiss_index,
    "faiss_HNSW": create_hnsw_faiss_index,
    "faiss_Flat": create_flat_faiss_index,
    "lancedb": create_lancedb_index,
}


def select_indexes(
    requested: Sequence[str],
) -> dict[str, Callable[[Path], AbstractVectorIndex]]:
    if not requested:
        return INDEX_FACTORIES

    canonical_by_lower = {
        index_name.lower(): index_name for index_name in INDEX_FACTORIES
    }
    selected: dict[str, Callable[[Path], AbstractVectorIndex]] = {}
    for name in requested:
        lowered = name.lower()
        canonical_name = canonical_by_lower.get(lowered)
        if canonical_name is None:
            raise ValueError(
                f"Unknown vector index '{name}'. Options: {sorted(INDEX_FACTORIES)}"
            )
        selected[canonical_name] = INDEX_FACTORIES[canonical_name]
    return selected


def benchmark_index(
    name: str,
    factory,
    embeddings: np.ndarray,
    pdf_names: Sequence[str],
    pdf_pages: Sequence[int],
    queries: np.ndarray,
    k: int,
    exact_kth_distances: np.ndarray,
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

    # Persist index artifacts when supported so disk size can be measured.
    if hasattr(index, "save_index"):
        try:
            index.save_index()
        except TypeError as exc:
            print(
                f"[WARN] Could not persist index '{name}' for size measurement: {exc}",
                file=sys.stderr,
            )

    index_size_mb = compute_directory_size_mb(index_dir)

    predicted_distances: list[np.ndarray] = []

    latencies: list[float] = []
    for query in queries:
        q_start = time.perf_counter()
        search_result = index.search(query, k)
        latencies.append(time.perf_counter() - q_start)

        # Validate interface output shape for metadata as well.
        extract_result_pairs(search_result, k)
        predicted_distances.append(extract_result_distances(search_result, k))

    total_query_time = sum(latencies) or 1e-9
    queries_per_sec = len(queries) / total_query_time
    avg_latency_ms = (statistics.mean(latencies) * 1000) if latencies else 0.0
    recall_at_k = compute_recall_at_k(
        predicted_distances, exact_kth_distances, k, atol=1e-2
    )

    ingest_embeddings_per_sec = (
        embeddings.shape[0] / ingest_seconds if ingest_seconds else float("inf")
    )
    return BenchmarkResult(
        name=name,
        embeddings=embeddings.shape[0],
        dimension=embeddings.shape[1],
        ingest_seconds=ingest_seconds,
        ingest_embeddings_per_sec=ingest_embeddings_per_sec,
        index_size_mb=index_size_mb,
        queries_per_sec=queries_per_sec,
        avg_latency_ms=avg_latency_ms,
        recall_at_k=recall_at_k,
    )


def format_results(results: Iterable[BenchmarkResult]) -> str:
    header = (
        f"{'Index':<12} {'Embeddings':>12} {'Dim':>6} {'Ingest (s)':>12} "
        f"{'Emb/s':>12} {'Index MB':>10} {'Queries/s':>12} {'Avg Latency (ms)':>18}\
              {'Recall@k':>10}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(
        f"{res.name:<12} {res.embeddings:>12} {res.dimension:>6} "
        f"{res.ingest_seconds:>12.4f} {res.ingest_embeddings_per_sec:>12.2f} "
        f"{res.index_size_mb:>10.2f} {res.queries_per_sec:>12.2f} \
            {res.avg_latency_ms:>18.2f} "
        f"{(f'{res.recall_at_k:.4f}' if res.recall_at_k is not None else 'n/a'):>10}"
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
        "--queries", type=int, default=10000, help="Number of random query vectors"
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
    exact_kth_distances: np.ndarray,
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
            exact_kth_distances,
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
    metadata_seed = rng.randrange(1 << 63)

    n_embeddings = args.embeddings + args.queries
    all_embeddings = generate_embeddings(n_embeddings, args.dim, embedding_seed)
    embeddings = all_embeddings[: args.embeddings]
    queries = all_embeddings[args.embeddings :]
    pdf_names, pdf_pages = generate_metadata(
        args.documents, args.embeddings, metadata_seed
    )

    k = max(1, min(args.k, args.embeddings))
    exact_kth_distances = compute_exact_topk_threshold(embeddings, queries, k)

    selected = select_indexes(args.indexes or [])
    index_root = args.index_root.resolve()
    index_root.mkdir(parents=True, exist_ok=True)

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
            exact_kth_distances,
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
