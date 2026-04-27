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


def _build_dataset(
    documents: int,
    pages_per_doc: int,
    dim: int,
    seed: int,
    selectivity_pct: int,
):
    rng = np.random.default_rng(seed)

    records = []
    digests = []
    pages = []
    vectors = []

    target_domain = "target.gov"
    non_target_domain = "other.gov"

    # Keep each scenario independent by assigning a fixed percentage of docs to
    # the filter-matching domain in this scenario's own dataset/index.
    match_count = int(round((documents * selectivity_pct) / 100.0))
    match_count = max(0, min(documents, match_count))

    for idx in range(documents):
        sub_domain = target_domain if idx < match_count else non_target_domain
        crawl_date = f"2024{(idx % 12) + 1:02d}{(idx % 28) + 1:02d}"

        pdf_name = f"{idx:040x}"
        records.append(
            {
                "crawl_url": f"https://{sub_domain}/report/{pdf_name}",
                "crawl_date": crawl_date,
                "digest": pdf_name,
                "sub_domain": sub_domain,
                "page_count": pages_per_doc,
                "s3_url": f"s3://govscape/{sub_domain}/{pdf_name}",
            }
        )

        base = rng.normal(size=dim).astype(np.float32)
        for page in range(pages_per_doc):
            noise = rng.normal(scale=0.05, size=dim).astype(np.float32)
            vec = base + noise
            digests.append(pdf_name)
            pages.append(page)
            vectors.append(vec)

    vector_array = np.asarray(vectors, dtype=np.float32)
    return records, digests, pages, vector_array


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
    return index


def _build_faiss_index(
    index_dir: Path,
    vector_digests: list[str],
    vector_pages: list[int],
    all_vectors: np.ndarray,
):
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Use FAISS flat index for deterministic postfilter benchmark behavior.
    faiss_index = FAISSIndex(index_dir.as_posix(), index_type="Flat")
    faiss_index.add_batch(all_vectors, vector_digests, vector_pages)
    return faiss_index


class PostFilteredIndex(HybridVectorMetadataIndex):
    # Override _choose_strategy to force postfiltering only for benchmarking purposes.
    def _choose_strategy(
        self, estimated_selectivity: float, target_results: int
    ) -> tuple[str, float, float]:
        # don't return real cost estimates since we won't use them
        return STRATEGY_POSTFILTER, -1, -1


class PreFilteredIndex(HybridVectorMetadataIndex):
    # Override _choose_strategy to force postfiltering only for benchmarking purposes.
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
    rows = []
    for scenario in scenarios:
        records, vector_pdf_names, vector_pages, all_vectors = _build_dataset(
            documents,
            pages_per_doc,
            dim,
            seed,
            scenario.selectivity_pct,
        )

        postFilteredIndex = PostFilteredIndex(
            metadata_index=_prepare_index(
                backend,
                work_dir / f"{backend}_{documents}_postfilter",
                records,
                vector_pdf_names,
                vector_pages,
                all_vectors,
                vector_batch_size,
            ),
            vector_index=_build_faiss_index(
                work_dir / f"{backend}_{documents}_faiss",
                vector_pdf_names,
                vector_pages,
                all_vectors,
            ),
        )

        preFilteredIndex = PreFilteredIndex(
            metadata_index=_prepare_index(
                backend,
                work_dir / f"{backend}_{documents}_prefilter",
                records,
                vector_pdf_names,
                vector_pages,
                all_vectors,
                vector_batch_size,
            ),
            vector_index=_build_faiss_index(
                work_dir / f"{backend}_{documents}_faiss",
                vector_pdf_names,
                vector_pages,
                all_vectors,
            ),
        )

        hybridIndex = HybridVectorMetadataIndex(
            metadata_index=_prepare_index(
                backend,
                work_dir / f"{backend}_{documents}_hybrid",
                records,
                vector_pdf_names,
                vector_pages,
                all_vectors,
                vector_batch_size,
            ),
            vector_index=_build_faiss_index(
                work_dir / f"{backend}_{documents}_faiss",
                vector_pdf_names,
                vector_pages,
                all_vectors,
            ),
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

        # Avoid open connection accumulation/locks across scenario tables.
        conn = getattr(preFilteredIndex.metadata_index, "conn", None)
        if conn is not None:
            conn.close()
            preFilteredIndex.conn = None

        conn = getattr(postFilteredIndex.metadata_index, "conn", None)
        if conn is not None:
            conn.close()
            postFilteredIndex.conn = None

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

    all_rows: list[BenchmarkRow] = []
    for backend in backends:
        for size in sizes:
            rows = run_benchmark(
                backend=backend,
                documents=size,
                pages_per_doc=args.pages_per_doc,
                dim=args.dim,
                queries=args.queries,
                k=args.k,
                work_dir=args.work_dir,
                seed=args.seed,
                vector_batch_size=args.vector_batch_size,
            )
            all_rows.extend(rows)

    print(format_results(all_rows))


if __name__ == "__main__":
    main()
