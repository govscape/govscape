# AI modified: 2026-02-21 d8ae3e4a
# AI modified: 2026-02-22 d8ae3e4a
"""Benchmark utilities for AbstractKeywordIndex implementations using real government documents.

Loads pages from a directory tree of the form:
    {txt_dir}/{digest}/{digest}_{pg_no}.txt

and queries from a plain-text file (one query per line), then benchmarks
various keyword index implementations for ingestion and query performance.

Example:
    poetry run python -m govscape.benchmarks.keyword_index_gov_benchmark \
        --txt-dir data/s3_mock/test-serving/txt \
        --queries-file data/queries/keyword_queries.txt \
        --num-queries 10000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

from govscape.indexing import (
    AbstractKeywordIndex,
    LanceDBKeywordIndex,
    SQLiteKeywordIndex,
    WhooshKeywordIndex,
)

# Registry of available keyword index implementations
INDEX_REGISTRY: Dict[str, Type[AbstractKeywordIndex]] = {
    "lancedb": LanceDBKeywordIndex,
    "sqlite": SQLiteKeywordIndex,
    "whoosh": WhooshKeywordIndex,
}

try:  # pragma: no cover - optional dependency
    from govscape.indexing import LuceneKeywordIndex  # type: ignore

    INDEX_REGISTRY["lucene"] = LuceneKeywordIndex
except Exception:  # pylint: disable=broad-except
    pass


def load_documents_from_txt_dir(txt_dir: Path) -> Tuple[List[str], List[str], List[int]]:
    """Walk {txt_dir}/{digest}/{digest}_{pg_no}.txt and return parallel lists.

    Returns:
        Tuple of (texts, pdf_names, page_numbers) where pdf_name is the digest
        and page_number is parsed from the filename suffix.
    """
    texts: List[str] = []
    pdf_names: List[str] = []
    pages: List[int] = []

    for digest_dir in sorted(txt_dir.iterdir()):
        if not digest_dir.is_dir():
            continue
        digest = digest_dir.name
        for txt_file in sorted(digest_dir.glob("*.txt")):
            stem = txt_file.stem  # e.g. "DIGEST_12"
            # Page number follows the last underscore
            try:
                page_num = int(stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            texts.append(text)
            pdf_names.append(digest)
            pages.append(page_num)

    return texts, pdf_names, pages


def load_queries_from_file(queries_file: Path) -> List[str]:
    """Read one query per non-empty line from a plain-text file."""
    queries: List[str] = []
    for line in queries_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            queries.append(line)
    return queries


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single index implementation."""
    name: str
    documents: int
    queries: int
    add_seconds: float
    ingest_docs_per_sec: float
    query_seconds: float
    queries_per_sec: float
    avg_query_latency_ms: float
    index_size_mb: float

    def as_row(self) -> Tuple[str, int, float, float, float, float, float]:
        return (
            self.name,
            self.documents,
            self.add_seconds,
            self.ingest_docs_per_sec,
            self.queries_per_sec,
            self.avg_query_latency_ms,
            self.index_size_mb,
        )


def benchmark_index(
    name: str,
    index_cls: Type[AbstractKeywordIndex],
    texts: Sequence[str],
    pdf_names: Sequence[str],
    pages: Sequence[int],
    queries: Sequence[str],
    k: int,
    index_root: Path,
) -> BenchmarkResult:
    """Benchmark a single keyword index implementation.
    
    Args:
        name: Name of the index implementation
        index_cls: Index class to instantiate
        texts: Document texts to index
        pdf_names: PDF file names for each document
        pages: Page numbers for each document
        queries: Queries to execute
        k: Number of results to retrieve per query
        index_root: Root directory for index storage
        
    Returns:
        BenchmarkResult with performance metrics
    """
    index_dir = index_root / name
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    index = index_cls(index_dir.as_posix())
    index.build_index()

    start_ingest = time.perf_counter()
    index.add_batch(texts, pdf_names, pages)
    index.save_index()
    ingest_duration = time.perf_counter() - start_ingest
    ingest_docs_per_sec = len(texts) / ingest_duration if ingest_duration else float("inf")

    index_size_mb = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file()) / (1024 * 1024)

    # Re-load the index to simulate real usage
    index = index_cls(index_dir.as_posix())
    index.load_index()

    query_latencies: List[float] = []
    for query in queries:
        q_start = time.perf_counter()
        index.search(query, k)
        query_latencies.append(time.perf_counter() - q_start)
    total_query_time = sum(query_latencies)
    queries_per_sec = len(queries) / total_query_time if total_query_time else float("inf")
    avg_latency_ms = statistics.mean(query_latencies) * 1000 if query_latencies else 0.0

    return BenchmarkResult(
        name=name,
        documents=len(texts),
        queries=len(queries),
        add_seconds=ingest_duration,
        ingest_docs_per_sec=ingest_docs_per_sec,
        query_seconds=total_query_time,
        queries_per_sec=queries_per_sec,
        avg_query_latency_ms=avg_latency_ms,
        index_size_mb=index_size_mb,
    )


def select_indexes(requested: Sequence[str]) -> Dict[str, Type[AbstractKeywordIndex]]:
    """Select which indexes to benchmark based on user input."""
    if not requested:
        return INDEX_REGISTRY
    selected: Dict[str, Type[AbstractKeywordIndex]] = {}
    for key in requested:
        lowered = key.lower()
        if lowered not in INDEX_REGISTRY:
            raise ValueError(
                f"Unknown keyword index implementation '{key}'. "
                f"Options: {sorted(INDEX_REGISTRY)}"
            )
        selected[lowered] = INDEX_REGISTRY[lowered]
    return selected


def format_results(results: Iterable[BenchmarkResult]) -> str:
    """Format benchmark results as a text table."""
    header = (
        f"{'Index':<18} {'Docs':>8} {'Ingest (s)':>12} {'Docs/s':>12} "
        f"{'Queries/s':>12} {'Avg Latency (ms)':>18} {'Size (MB)':>12}"
    )
    lines = [header, "-" * len(header)]
    for res in results:
        lines.append(
            f"{res.name:<18} {res.documents:>8} {res.add_seconds:>12.4f} "
            f"{res.ingest_docs_per_sec:>12.2f} "
            f"{res.queries_per_sec:>12.2f} {res.avg_query_latency_ms:>18.2f} "
            f"{res.index_size_mb:>12.2f}"
        )
    return "\n".join(lines)


def write_csv(results: Iterable[BenchmarkResult], path: Path) -> None:
    """Write benchmark results to a CSV file."""
    result_list = list(results)
    if not result_list:
        return
    column_names = [f.name for f in fields(BenchmarkResult)]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=column_names)
        writer.writeheader()
        for res in result_list:
            writer.writerow({
                "name": res.name,
                "documents": res.documents,
                "queries": res.queries,
                "add_seconds": f"{res.add_seconds:.4f}",
                "ingest_docs_per_sec": f"{res.ingest_docs_per_sec:.2f}",
                "query_seconds": f"{res.query_seconds:.4f}",
                "queries_per_sec": f"{res.queries_per_sec:.2f}",
                "avg_query_latency_ms": f"{res.avg_query_latency_ms:.2f}",
                "index_size_mb": f"{res.index_size_mb:.2f}",
            })


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark keyword index implementations using real government documents"
    )
    parser.add_argument(
        "--txt-dir",
        type=Path,
        required=True,
        help=(
            "Root directory containing per-digest subdirectories of page txt files. "
            "Expected layout: {txt_dir}/{digest}/{digest}_{pg_no}.txt"
        ),
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        required=True,
        help="Plain-text file with one query per line.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to request per query",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("./.keyword_index_gov_benchmark"),
        help="Directory where benchmark indices will be stored",
    )
    parser.add_argument(
        "--indexes",
        nargs="*",
        default=None,
        help=f"Subset of indexes to run ({', '.join(sorted(INDEX_REGISTRY))})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write results to this CSV file in addition to stdout.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10_000,
        help="Total number of queries to run, sampling with replacement when the"
             " file has fewer entries (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling queries (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the government document benchmark."""
    args = parse_args(argv or sys.argv[1:])

    index_root = args.index_root.resolve()
    index_root.mkdir(parents=True, exist_ok=True)

    txt_dir = args.txt_dir.resolve()
    if not txt_dir.exists():
        print(f"[ERROR] txt directory not found: {txt_dir}", file=sys.stderr)
        return 1

    queries_file = args.queries_file.resolve()
    if not queries_file.exists():
        print(f"[ERROR] Queries file not found: {queries_file}", file=sys.stderr)
        return 1

    # Load documents from txt directory tree
    print(f"[INFO] Loading documents from {txt_dir}...", file=sys.stderr)
    texts, pdf_names, pages = load_documents_from_txt_dir(txt_dir)
    if not texts:
        print("[ERROR] No documents loaded", file=sys.stderr)
        return 1
    print(f"[INFO] Loaded {len(texts)} pages from {len(set(pdf_names))} documents", file=sys.stderr)

    # Load queries from file
    print(f"[INFO] Loading queries from {queries_file}...", file=sys.stderr)
    base_queries = load_queries_from_file(queries_file)
    if not base_queries:
        print("[ERROR] No queries loaded", file=sys.stderr)
        return 1
    print(f"[INFO] Loaded {len(base_queries)} unique queries", file=sys.stderr)

    # Expand to the requested trial count by sampling with replacement
    rng = random.Random(args.seed)
    if args.num_queries <= len(base_queries):
        queries = rng.sample(base_queries, args.num_queries)
    else:
        queries = base_queries + rng.choices(base_queries, k=args.num_queries - len(base_queries))
    print(f"[INFO] Running {len(queries)} queries (seed={args.seed})", file=sys.stderr)

    # Run benchmarks
    selected = select_indexes(args.indexes or [])
    results: List[BenchmarkResult] = []
    for name, index_cls in selected.items():
        print(f"[INFO] Benchmarking index: {name}", file=sys.stderr)
        try:
            result = benchmark_index(
                name=name,
                index_cls=index_cls,
                texts=texts,
                pdf_names=pdf_names,
                pages=pages,
                queries=queries,
                k=args.k,
                index_root=index_root,
            )
            results.append(result)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] Skipping {name} due to error: {exc}", file=sys.stderr)

    if not results:
        print("No successful benchmarks were recorded.", file=sys.stderr)
        return 1

    print(format_results(results))
    if args.csv is not None:
        write_csv(results, args.csv.resolve())
        print(f"[INFO] CSV written to {args.csv.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
