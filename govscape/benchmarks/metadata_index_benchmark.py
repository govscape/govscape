# AI modified: 2026-03-08 f62d40b8
# AI modified: 2026-03-08 4efba197
# AI modified: 2026-03-08 4efba197
# AI modified: 2026-03-08 4efba197
# AI modified: 2026-03-09 4efba197
"""Benchmark utilities for AbstractMetadataIndex implementations.

This script generates synthetic metadata records on the fly, feeds them
through available metadata index implementations, and reports basic throughput
metrics for both ingestion and querying under three filter scenarios:

  - no_filter       : look up pdf_names with no additional constraints
  - domain_filter   : filter by sub_domain only
  - date_filter     : filter by crawled_after and crawled_before only
  - all_filters: filter by sub_domain, crawled_after, and crawled_before

Example:
    poetry run python -m govscape.benchmarks.metadata_index_benchmark \\
        --documents 5000 --queries 200
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from govscape.indexing import (
    AbstractMetadataIndex,
    DuckDBMetadataIndex,
    SQLiteMetadataIndex,
)
from govscape.query import EqualityPredicate, Predicate, RangePredicate

INDEX_REGISTRY: dict[str, type[AbstractMetadataIndex]] = {
    "sqlite": SQLiteMetadataIndex,
    "duckdb": DuckDBMetadataIndex,
}

# Synthetic sub-domains that documents are drawn from.
_SUB_DOMAINS = [
    "epa.gov",
    "energy.gov",
    "nasa.gov",
    "cdc.gov",
    "nih.gov",
    "nsf.gov",
    "defense.gov",
    "state.gov",
]


def _random_date(rng: random.Random) -> str:
    """Return a random date string in YYYYMMDD format."""
    year = rng.randint(2018, 2025)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}{month:02d}{day:02d}"


def generate_metadata(
    num_docs: int,
    seed: int,
) -> list[dict]:
    """Generate ``num_docs`` synthetic metadata dicts."""
    rng = random.Random(seed)
    records: list[dict] = []
    for idx in range(num_docs):
        sub_domain = rng.choice(_SUB_DOMAINS)
        crawl_date = _random_date(rng)
        pdf_name = f"doc_{idx:06d}.pdf"
        records.append(
            {
                "crawl_url": f"https://{sub_domain}/reports/{pdf_name}",
                "crawl_date": crawl_date,
                "pdf_name": pdf_name,
                "sub_domain": sub_domain,
                "page_count": rng.randint(1, 200),
                "s3_url": f"s3://govscape/{sub_domain}/{pdf_name}",
            }
        )
    return records


# Type alias for a single query: (pdf_names, filter_dict)
IndexQuery = tuple[list[str], list[Predicate]]


def generate_query_batches(
    num_queries: int,
    records: Sequence[dict],
    seed: int,
    batch_size: int = 10,
) -> dict[str, list[IndexQuery]]:
    """
    Generate query tuples split by scenario.

    Returns a dict with keys:
      - no_filter         : filter is None
      - domain_filter     : filter contains sub_domain only
      - date_filter       : filter contains crawled_after and crawled_before only
      - all_filters: filter contains sub_domain, crawled_after, and crawled_before

    Each scenario gets an equal share of ``num_queries``.
    """
    rng = random.Random(seed)
    pdf_names_pool = [r["pdf_name"] for r in records]

    scenario_count = num_queries // 4
    remainder = num_queries - scenario_count * 4

    def _sample_names() -> list[str]:
        k = min(batch_size, len(pdf_names_pool))
        return rng.sample(pdf_names_pool, k=k)

    def _random_date_range() -> tuple[str, str]:
        year_after = rng.randint(2018, 2021)
        year_before = rng.randint(year_after + 1, 2025)
        after = f"{year_after:04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        before = f"{year_before:04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        return after, before

    no_filter: list[IndexQuery] = [(_sample_names(), []) for _ in range(scenario_count)]

    domain_filter: list[IndexQuery] = [
        (_sample_names(), [EqualityPredicate("sub_domain", rng.choice(_SUB_DOMAINS))])
        for _ in range(scenario_count)
    ]

    date_filter: list[IndexQuery] = []
    for _ in range(scenario_count):
        after, before = _random_date_range()
        date_filter.append(
            (
                _sample_names(),
                [RangePredicate("crawl_date", min_val=after, max_val=before)],
            )
        )

    all_filters: list[IndexQuery] = []
    for _ in range(scenario_count + remainder):
        after, before = _random_date_range()
        all_filters.append(
            (
                _sample_names(),
                [
                    EqualityPredicate("sub_domain", rng.choice(_SUB_DOMAINS)),
                    RangePredicate("crawl_date", min_val=after, max_val=before),
                ],
            )
        )

    return {
        "no_filter": no_filter,
        "domain_filter": domain_filter,
        "date_filter": date_filter,
        "all_filters": all_filters,
    }


@dataclass
class BenchmarkResult:
    name: str
    documents: int
    queries: int
    add_seconds: float
    ingest_docs_per_sec: float
    index_size_bytes: int
    # Per-scenario timings
    no_filter_queries: int
    no_filter_qps: float
    no_filter_avg_ms: float
    domain_filter_queries: int
    domain_filter_qps: float
    domain_filter_avg_ms: float
    date_filter_queries: int
    date_filter_qps: float
    date_filter_avg_ms: float
    all_filters_queries: int
    all_filters_qps: float
    all_filters_avg_ms: float


def _run_queries(
    index: AbstractMetadataIndex,
    queries: list[IndexQuery],
) -> tuple[float, float, float]:
    """Run all queries, return (total_s, q/s, avg_latency_ms)."""
    if not queries:
        return 0.0, 0.0, 0.0
    latencies: list[float] = []
    for pdf_names, filt in queries:
        t0 = time.perf_counter()
        index.search(pdf_names, filt)
        latencies.append(time.perf_counter() - t0)
    total = sum(latencies)
    qps = len(latencies) / total if total else float("inf")
    avg_ms = statistics.mean(latencies) * 1000
    return total, qps, avg_ms


def benchmark_index(
    name: str,
    index_cls: type[AbstractMetadataIndex],
    records: Sequence[dict],
    query_batches: dict[str, list[IndexQuery]],
    index_root: Path,
) -> BenchmarkResult:
    index_dir = index_root / name
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index = index_cls(index_dir.as_posix())
    index.build_index()

    t0 = time.perf_counter()
    index.add_batch(list(records))
    index.save_index()
    ingest_duration = time.perf_counter() - t0
    ingest_dps = len(records) / ingest_duration if ingest_duration else float("inf")

    # Re-load to simulate real serving usage.
    index = index_cls(index_dir.as_posix())
    index.load_index()

    _, nf_qps, nf_ms = _run_queries(index, query_batches["no_filter"])
    _, dmf_qps, dmf_ms = _run_queries(index, query_batches["domain_filter"])
    _, df_qps, df_ms = _run_queries(index, query_batches["date_filter"])
    _, ddf_qps, ddf_ms = _run_queries(index, query_batches["all_filters"])

    index_size = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())
    total_queries = sum(len(b) for b in query_batches.values())
    return BenchmarkResult(
        name=name,
        documents=len(records),
        queries=total_queries,
        add_seconds=ingest_duration,
        ingest_docs_per_sec=ingest_dps,
        index_size_bytes=index_size,
        no_filter_queries=len(query_batches["no_filter"]),
        no_filter_qps=nf_qps,
        no_filter_avg_ms=nf_ms,
        domain_filter_queries=len(query_batches["domain_filter"]),
        domain_filter_qps=dmf_qps,
        domain_filter_avg_ms=dmf_ms,
        date_filter_queries=len(query_batches["date_filter"]),
        date_filter_qps=df_qps,
        date_filter_avg_ms=df_ms,
        all_filters_queries=len(query_batches["all_filters"]),
        all_filters_qps=ddf_qps,
        all_filters_avg_ms=ddf_ms,
    )


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def format_results(results: Iterable[BenchmarkResult]) -> str:
    ci, cd, cn, cds, csz, cs, cq, cm = 14, 6, 10, 8, 8, 13, 9, 10
    header = (
        f"{'Index':<{ci}} {'Docs':>{cd}} {'Ingest(s)':>{cn}} {'Docs/s':>{cds}}"
        f" {'Size':>{csz}}  {'Scenario':<{cs}} {'Q/s':>{cq}} {'Lat(ms)':>{cm}}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    scenarios = [
        ("no_filter", "no_filter_qps", "no_filter_avg_ms"),
        ("domain_filter", "domain_filter_qps", "domain_filter_avg_ms"),
        ("date_filter", "date_filter_qps", "date_filter_avg_ms"),
        ("all_filters", "all_filters_qps", "all_filters_avg_ms"),
    ]

    for res in results:
        prefix = (
            f"{res.name:<{ci}} {res.documents:>{cd}} {res.add_seconds:>{cn}.4f}"
            f" {res.ingest_docs_per_sec:>{cds}.0f}"
            f" {_fmt_size(res.index_size_bytes):>{csz}}"
        )
        blank = " " * len(prefix)
        for i, (label, qps_attr, ms_attr) in enumerate(scenarios):
            qps = getattr(res, qps_attr)
            avg_ms = getattr(res, ms_attr)
            lead = prefix if i == 0 else blank
            lines.append(f"{lead}  {label:<{cs}} {qps:>{cq}.1f} {avg_ms:>{cm}.2f}")
        lines.append(sep)
    return "\n".join(lines)


def select_indexes(requested: Sequence[str]) -> dict[str, type[AbstractMetadataIndex]]:
    if not requested:
        return INDEX_REGISTRY
    selected: dict[str, type[AbstractMetadataIndex]] = {}
    for key in requested:
        lowered = key.lower()
        if lowered not in INDEX_REGISTRY:
            raise ValueError(
                f"Unknown metadata index '{key}'. Options: {sorted(INDEX_REGISTRY)}"
            )
        selected[lowered] = INDEX_REGISTRY[lowered]
    return selected


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark metadata index implementations"
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=5000,
        help="Number of synthetic metadata records to generate",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=300,
        help="Total number of queries (divided equally across filter scenarios)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of pdf_names to look up per query",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("./.metadata_index_benchmark"),
        help="Directory where benchmark indices will be stored",
    )
    parser.add_argument(
        "--indexes",
        nargs="*",
        default=None,
        help=f"Subset of indexes to run ({', '.join(sorted(INDEX_REGISTRY))})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for reproducibility",
    )
    return parser.parse_args(argv)


def try_benchmark(
    name: str,
    index_cls: type[AbstractMetadataIndex],
    records: Sequence[dict],
    query_batches: dict[str, list[IndexQuery]],
    index_root: Path,
) -> BenchmarkResult | None:
    try:
        return benchmark_index(
            name=name,
            index_cls=index_cls,
            records=records,
            query_batches=query_batches,
            index_root=index_root,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Skipping {name} due to error: {exc}", file=sys.stderr)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    random.seed(args.seed)
    index_root = args.index_root.resolve()
    index_root.mkdir(parents=True, exist_ok=True)

    doc_seed = random.randrange(1 << 63)
    query_seed = random.randrange(1 << 63)

    records = generate_metadata(args.documents, seed=doc_seed)
    query_batches = generate_query_batches(
        args.queries,
        records,
        seed=query_seed,
        batch_size=args.batch_size,
    )

    selected = select_indexes(args.indexes or [])
    results: list[BenchmarkResult] = []
    for name, index_cls in selected.items():
        result = try_benchmark(
            name=name,
            index_cls=index_cls,
            records=records,
            query_batches=query_batches,
            index_root=index_root,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("No successful benchmarks were recorded.", file=sys.stderr)
        return 1

    print(format_results(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
