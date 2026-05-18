# AI modified
"""Benchmark hybrid vector metadata index against prefilter and postfilter strategies.

It runs both strategies across configurable corpus sizes and with both
SQLite and DuckDB metadata indexes.

Example:
    poetry run python benchmarks/hybrid_vector_index_benchmark.py \
        --sizes 1000,5000,20000 --queries 20 --backends sqlite,duckdb
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from govscape.indexing import DuckDBMetadataIndex, SQLiteMetadataIndex
from govscape.indexing.hybrid import (
    STRATEGY_POSTFILTER,
    STRATEGY_PREFILTER,
    HybridVectorMetadataIndex,
)
from govscape.indexing.vector import FAISSIndex
from govscape.query import EqualityPredicate, Predicate


@dataclass
class Scenario:
    name: str
    selectivity_pct: int
    predicates: list[Predicate]


@dataclass
class BenchmarkRow:
    backend: str
    documents: int
    scenario: str
    prefilter_avg_ms: float
    postfilter_avg_ms: float
    hybrid_avg_ms: float
    speedup_post_over_pre: float


def _parse_csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_vectors(
    documents: int,
    pages_per_doc: int,
    dim: int,
    seed: int,
):
    """Generate page-level vectors. Independent of selectivity."""
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    pdf_names = [f"{idx:040x}" for idx in range(documents)]
    bases = rng.normal(size=(documents, dim)).astype(np.float32)
    noise = rng.normal(scale=0.05, size=(documents, pages_per_doc, dim)).astype(
        np.float32
    )
    vector_array = (bases[:, np.newaxis, :] + noise).reshape(
        documents * pages_per_doc, dim
    )
    digests = np.repeat(pdf_names, pages_per_doc).tolist()
    pages = np.tile(np.arange(pages_per_doc), documents).tolist()
    print(
        f"Built {len(digests)} vectors ({documents} docs × {pages_per_doc} pages) "
        f"in {time.perf_counter() - start:.2f}s"
    )
    return pdf_names, digests, pages, vector_array


def _build_records(
    documents: int,
    pages_per_doc: int,
    pdf_names: list[str],
    selectivity_pct: int,
):
    """Build metadata records for a given selectivity. Cheap — no RNG."""
    target_domain = "target.gov"
    non_target_domain = "other.gov"
    match_count = max(
        0, min(documents, int(round((documents * selectivity_pct) / 100.0)))
    )
    sub_domains = [
        target_domain if idx < match_count else non_target_domain
        for idx in range(documents)
    ]
    crawl_dates = [
        f"2024{(idx % 12) + 1:02d}{(idx % 28) + 1:02d}" for idx in range(documents)
    ]
    return [
        {
            "crawl_url": f"https://{sub_domains[i]}/report/{pdf_names[i]}",
            "crawl_date": crawl_dates[i],
            "digest": pdf_names[i],
            "sub_domain": sub_domains[i],
            "page_count": pages_per_doc,
            "s3_url": f"s3://govscape/{sub_domains[i]}/{pdf_names[i]}",
        }
        for i in range(documents)
    ]


def _create_metadata_index(
    backend: str,
    index_dir: Path,
):
    if backend == "sqlite":
        return SQLiteMetadataIndex(index_dir.as_posix())
    if backend == "duckdb":
        return DuckDBMetadataIndex(index_dir.as_posix())
    raise ValueError(f"Unsupported backend: {backend}")


def _prepare_index(
    backend: str,
    index_dir: Path,
    records,
    vector_digests,
    vector_pages,
    vector_array,
    vector_batch_size: int,
):
    # time the index creation
    print(
        f"Preparing {backend} with {len(records)} docs, {len(vector_digests)} vectors"
    )

    start_time = time.perf_counter()

    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index = _create_metadata_index(backend, index_dir)
    index.build_index()
    index.add_batch(records)

    for start in range(0, len(vector_digests), vector_batch_size):
        end = min(start + vector_batch_size, len(vector_digests))
        index.upsert_vectors(
            "textual",
            vector_array[start:end],
            vector_digests[start:end],
            vector_pages[start:end],
        )

    index.save_index()

    print(
        f"Built {backend} index with {len(records)} records and vectors in "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    return index


def _build_faiss_index(
    index_dir: Path,
    vector_digests: list[str],
    vector_pages: list[int],
    all_vectors: np.ndarray,
):
    # time the FAISS index creation
    print(f"Building FAISS index with {len(vector_digests)} vectors")
    start = time.perf_counter()

    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_index = FAISSIndex(index_dir.as_posix())

    chunk = 8192 * 16
    n = all_vectors.shape[0]
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        faiss_index.add_batch(all_vectors[i:j], vector_digests[i:j], vector_pages[i:j])
    print(
        f"Built FAISS index with {len(vector_digests)} vectors in "
        f"{time.perf_counter() - start:.2f} seconds"
    )

    return faiss_index


class PostFilteredIndex(HybridVectorMetadataIndex):
    # Override _choose_strategy to force postfiltering only for benchmarking purposes.
    def _choose_strategy(
        self, estimated_selectivity: float, target_results: int
    ) -> tuple[str, float, float]:
        # don't return real cost estimates since we won't use them
        return STRATEGY_POSTFILTER, -1, -1


class PreFilteredIndex(HybridVectorMetadataIndex):
    # Override _choose_strategy to force prefiltering only for benchmarking purposes.
    def _choose_strategy(
        self, estimated_selectivity: float, target_results: int
    ) -> tuple[str, float, float]:
        # don't return real cost estimates since we won't use them
        return STRATEGY_PREFILTER, -1, -1


def eq(key: str, value: str) -> EqualityPredicate:
    return EqualityPredicate(key, value)


def run_benchmark(
    backend: str,
    documents: int,
    pages_per_doc: int,
    dim: int,
    queries: int,
    k: int,
    work_dir: Path,
    seed: int,
    vector_batch_size: int,
) -> list[BenchmarkRow]:
    scenarios = [
        Scenario("selective_1pct", 1, [eq("sub_domain", "target.gov")]),
        Scenario("selective_2pct", 2, [eq("sub_domain", "target.gov")]),
        Scenario("selective_5pct", 5, [eq("sub_domain", "target.gov")]),
        Scenario("selective_10pct", 10, [eq("sub_domain", "target.gov")]),
        Scenario("selective_50pct", 50, [eq("sub_domain", "target.gov")]),
        Scenario("selective_90pct", 90, [eq("sub_domain", "target.gov")]),
        Scenario("selective_100pct", 100, [eq("sub_domain", "target.gov")]),
    ]

    rng = random.Random(seed + 17)

    # Vectors are independent of selectivity — build once and share across scenarios.
    pdf_names, vector_digests, vector_pages, all_vectors = _build_vectors(
        documents, pages_per_doc, dim, seed
    )
    shared_faiss = _build_faiss_index(
        work_dir / f"{backend}_{documents}_faiss",
        vector_digests,
        vector_pages,
        all_vectors,
    )

    rows = []
    for scenario in scenarios:
        records = _build_records(
            documents, pages_per_doc, pdf_names, scenario.selectivity_pct
        )

        shared_meta = _prepare_index(
            backend,
            work_dir / f"{backend}_{documents}_{scenario.name}",
            records,
            vector_digests,
            vector_pages,
            all_vectors,
            vector_batch_size,
        )

        postFilteredIndex = PostFilteredIndex(
            metadata_index=shared_meta,
            vector_index=shared_faiss,
        )
        preFilteredIndex = PreFilteredIndex(
            metadata_index=shared_meta,
            vector_index=shared_faiss,
        )
        hybridIndex = HybridVectorMetadataIndex(
            metadata_index=shared_meta,
            vector_index=shared_faiss,
        )

        pre_times = []
        post_times = []
        hybrid_times = []

        for _ in range(queries):
            q_idx = rng.randint(0, all_vectors.shape[0] - 1)
            query_vector = all_vectors[q_idx]

            t0 = time.perf_counter()
            preFilteredIndex.search(query_vector, scenario.predicates, k)
            pre_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            postFilteredIndex.search(query_vector, scenario.predicates, k)
            post_times.append(time.perf_counter() - t1)

            t2 = time.perf_counter()
            hybridIndex.search(query_vector, scenario.predicates, k)
            hybrid_times.append(time.perf_counter() - t2)

        pre_ms = statistics.mean(pre_times) * 1000
        post_ms = statistics.mean(post_times) * 1000
        hybrid_ms = statistics.mean(hybrid_times) * 1000
        speedup = post_ms / pre_ms if pre_ms > 0 else float("inf")

        rows.append(
            BenchmarkRow(
                backend=backend,
                documents=documents,
                scenario=scenario.name,
                prefilter_avg_ms=pre_ms,
                postfilter_avg_ms=post_ms,
                hybrid_avg_ms=hybrid_ms,
                speedup_post_over_pre=speedup,
            )
        )

        # Close the shared connection once (all three wrappers point at same object).
        conn = getattr(shared_meta, "conn", None)
        if conn is not None:
            conn.close()
            shared_meta.conn = None

    return rows


def format_results(rows: Iterable[BenchmarkRow]) -> str:
    header = (
        f"{'Backend':<10} {'Documents':>10} {'Scenario':<16} "
        f"{'Prefilter (ms)':>14} {'Postfilter (ms)':>15} "
        f"{'Hybrid (ms)':>15} {'Post/Pre':>10}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(
        [
            f"{row.backend:<10} {row.documents:>10} {row.scenario:<16} "
            f"{row.prefilter_avg_ms:>14.3f} {row.postfilter_avg_ms:>15.3f} "
            f"{row.hybrid_avg_ms:>15.3f} {row.speedup_post_over_pre:>10.3f}"
            for row in rows
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prefilter vs postfilter across corpus sizes/backends"
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="1000,5000,20000",
        help="Comma-separated document counts.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default="sqlite,duckdb",
        help="Comma-separated metadata backends: sqlite,duckdb.",
    )
    parser.add_argument("--pages-per-doc", type=int, default=4)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--vector-batch-size",
        type=int,
        default=50000,
        help="Batch size when writing vectors into metadata index.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/govscape_filter_strategy_bench"),
        help="Temporary directory for index artifacts.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sizes = _parse_csv_ints(args.sizes)
    backends = _parse_csv_strings(args.backends)

    common = {
        "pages_per_doc": args.pages_per_doc,
        "dim": args.dim,
        "queries": args.queries,
        "k": args.k,
        "work_dir": args.work_dir,
        "seed": args.seed,
        "vector_batch_size": args.vector_batch_size,
    }

    tasks = [(b, s) for b in backends for s in sizes]
    all_rows: list[BenchmarkRow] = []

    with ProcessPoolExecutor() as pool:
        futures = {
            pool.submit(run_benchmark, backend=b, documents=s, **common): (b, s)
            for b, s in tasks
        }
        for fut in as_completed(futures):
            all_rows.extend(fut.result())

    # Sort for stable output order regardless of completion order.
    all_rows.sort(
        key=lambda r: (
            r.backend,
            r.documents,
            int(r.scenario.removeprefix("selective_").removesuffix("pct")),
        )
    )
    print(format_results(all_rows))


if __name__ == "__main__":
    main()
