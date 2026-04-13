# AI modified: 2026-04-12 18:35:40 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 18:49:32 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 21:52:17 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 22:07:09 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 22:25:10 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 22:36:39 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 22:41:29 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 23:02:34 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 23:05:04 fddc6344a807a84c8b9161bd3ffeded5153c5e27
"""Benchmark prefilter vs postfilter retrieval strategies.

This benchmark compares two retrieval paths that mirror the server behavior:

- postfilter: global nearest-neighbor search then metadata filtering
- prefilter : metadata filtering first, then nearest-neighbor over candidates

It runs both strategies across configurable corpus sizes and with both
SQLite and DuckDB metadata indexes.

Example:
    poetry run python benchmarks/filter_strategy_benchmark.py \
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

from govscape.indexing import DuckDBMetadataIndex, FAISSIndex, SQLiteMetadataIndex


@dataclass
class Scenario:
    name: str
    selectivity_pct: int
    filter_dict: dict[str, str]


@dataclass
class BenchmarkRow:
    backend: str
    documents: int
    scenario: str
    prefilter_avg_ms: float
    postfilter_avg_ms: float
    speedup_post_over_pre: float


def _parse_csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _deduplicate_responses(distances, names, pages):
    seen = set()
    unique_distances = []
    unique_names = []
    unique_pages = []
    for distance, name, page in zip(distances, names, pages, strict=False):
        if name not in seen:
            seen.add(name)
            unique_distances.append(distance)
            unique_names.append(name)
            unique_pages.append(page)
    return unique_distances, unique_names, unique_pages


def _build_dataset(
    documents: int,
    pages_per_doc: int,
    dim: int,
    seed: int,
    selectivity_pct: int,
):
    rng = np.random.default_rng(seed)

    records = []
    pdf_names = []
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
                "pdf_name": pdf_name,
                "sub_domain": sub_domain,
                "page_count": pages_per_doc,
                "s3_url": f"s3://govscape/{sub_domain}/{pdf_name}",
            }
        )

        base = rng.normal(size=dim).astype(np.float32)
        for page in range(pages_per_doc):
            noise = rng.normal(scale=0.05, size=dim).astype(np.float32)
            vec = base + noise
            pdf_names.append(pdf_name)
            pages.append(page)
            vectors.append(vec)

    vector_array = np.asarray(vectors, dtype=np.float32)
    return records, pdf_names, pages, vector_array


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
    vector_pdf_names,
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

    for start in range(0, len(vector_pdf_names), vector_batch_size):
        end = min(start + vector_batch_size, len(vector_pdf_names))
        index.upsert_vectors_batch(
            "textual",
            vector_pdf_names[start:end],
            vector_pages[start:end],
            vector_array[start:end],
        )

    index.save_index()
    return index


def _build_faiss_index(
    index_dir: Path,
    vector_pdf_names: list[str],
    vector_pages: list[int],
    all_vectors: np.ndarray,
):
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Use FAISS flat index for deterministic postfilter benchmark behavior.
    faiss_index = FAISSIndex(index_dir.as_posix(), index_type="Flat")
    faiss_index.add_batch(all_vectors, vector_pdf_names, vector_pages)
    return faiss_index


def _postfilter_search_count(
    faiss_index,
    metadata_index,
    query_vector: np.ndarray,
    filter_dict: dict[str, str] | None,
    k: int,
):
    current_k = k * 2
    search_results_count = 0
    old_results_found = -1
    total_entries = faiss_index.total_entries()

    while search_results_count < k:
        top_k = min(current_k, int(total_entries))
        D, pdf_names, pdf_pages = faiss_index.search(query_vector, top_k)

        D, pdf_names, pdf_pages = _deduplicate_responses(D, pdf_names, pdf_pages)

        metadata = metadata_index.search(pdf_names, filter_dict)
        search_results_count = 0
        for name in pdf_names:
            if metadata.get(name):
                search_results_count += 1

        if current_k > min(100000, total_entries):
            break

        if search_results_count >= k:
            break

        results_found = search_results_count
        if results_found == old_results_found and (len(filter_dict or {}) == 0):
            break
        old_results_found = results_found

        current_k *= 2

    return search_results_count


def _prefilter_search_count(
    metadata_index,
    query_vector: np.ndarray,
    filter_dict: dict[str, str] | None,
    k: int,
):
    pdf_page_counts = metadata_index.get_filtered_pdf_page_counts(filter_dict)
    if len(pdf_page_counts) == 0:
        return 0

    candidate_vectors = metadata_index.get_vectors_for_pdf_page_counts(
        "textual",
        pdf_page_counts,
    )
    if len(candidate_vectors) == 0:
        return 0

    best_by_pdf = {}
    for pdf_name, page_vectors in candidate_vectors.items():
        best_for_pdf = None
        for page_num, page_embedding in page_vectors:
            if page_embedding.ndim > 1:
                page_embedding = page_embedding.reshape(-1)
            distance = float(np.sum((page_embedding - query_vector) ** 2))
            if best_for_pdf is None or distance < best_for_pdf[0]:
                best_for_pdf = (distance, str(page_num))

        if best_for_pdf is not None:
            best_by_pdf[pdf_name] = best_for_pdf

    if len(best_by_pdf) == 0:
        return 0

    ranked = sorted(best_by_pdf.items(), key=lambda x: x[1][0])
    ranked = ranked[:k]
    return len(ranked)


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
        Scenario("selective_1pct", 1, {"sub_domain": "target.gov"}),
        Scenario("selective_2pct", 2, {"sub_domain": "target.gov"}),
        Scenario("selective_5pct", 5, {"sub_domain": "target.gov"}),
        Scenario("selective_10pct", 10, {"sub_domain": "target.gov"}),
        Scenario("selective_50pct", 50, {"sub_domain": "target.gov"}),
        Scenario("selective_90pct", 90, {"sub_domain": "target.gov"}),
        Scenario("selective_100pct", 100, {"sub_domain": "target.gov"}),
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

        index = _prepare_index(
            backend,
            work_dir / f"{backend}_{documents}_{scenario.name}",
            records,
            vector_pdf_names,
            vector_pages,
            all_vectors,
            vector_batch_size,
        )
        faiss_index = _build_faiss_index(
            work_dir / f"faiss_{backend}_{documents}_{scenario.name}",
            vector_pdf_names,
            vector_pages,
            all_vectors,
        )

        pre_times = []
        post_times = []

        for _ in range(queries):
            q_idx = rng.randint(0, all_vectors.shape[0] - 1)
            query_vector = all_vectors[q_idx]

            t0 = time.perf_counter()
            _prefilter_search_count(index, query_vector, scenario.filter_dict, k)
            pre_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            _postfilter_search_count(
                faiss_index,
                index,
                query_vector,
                scenario.filter_dict,
                k,
            )
            post_times.append(time.perf_counter() - t1)

        pre_ms = statistics.mean(pre_times) * 1000
        post_ms = statistics.mean(post_times) * 1000
        speedup = post_ms / pre_ms if pre_ms > 0 else float("inf")

        rows.append(
            BenchmarkRow(
                backend=backend,
                documents=documents,
                scenario=scenario.name,
                prefilter_avg_ms=pre_ms,
                postfilter_avg_ms=post_ms,
                speedup_post_over_pre=speedup,
            )
        )

        # Avoid open connection accumulation/locks across scenario tables.
        conn = getattr(index, "conn", None)
        if conn is not None:
            conn.close()
            index.conn = None

    return rows


def format_results(rows: Iterable[BenchmarkRow]) -> str:
    header = (
        f"{'Backend':<10} {'Documents':>10} {'Scenario':<16} "
        f"{'Prefilter (ms)':>14} {'Postfilter (ms)':>15} {'Post/Pre':>10}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(
        [
            f"{row.backend:<10} {row.documents:>10} {row.scenario:<16} "
            f"{row.prefilter_avg_ms:>14.3f} {row.postfilter_avg_ms:>15.3f} "
            f"{row.speedup_post_over_pre:>10.3f}"
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
