"""Benchmark utilities for AbstractKeywordIndex implementations.

This script generates synthetic lorem-ipsum style documents and queries on the
fly, feeds them through every available keyword index implementation, and
reports basic throughput metrics for both ingestion and querying.

Example:
    poetry run python -m govscape.benchmarks.keyword_index_benchmark \
        --documents 2000 --words-per-document 120 --queries 200
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
from multiprocessing import Pool, cpu_count
from pathlib import Path

from govscape.indexing import (
    AbstractKeywordIndex,
    LanceDBKeywordIndex,
    SQLiteKeywordIndex,
    WhooshKeywordIndex,
)

# Optional imports live behind try/except so the benchmark still runs even if a
# dependency is unavailable in the active environment.
INDEX_REGISTRY: dict[str, type[AbstractKeywordIndex]] = {
    "lancedb": LanceDBKeywordIndex,
    "sqlite": SQLiteKeywordIndex,
    "whoosh": WhooshKeywordIndex,
}

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def generate_vocabulary(size: int, seed: int) -> list[str]:
    if size <= 0:
        raise ValueError("Vocabulary size must be positive")
    rng = random.Random(seed)
    vocab: list[str] = []
    for idx in range(size):
        length = rng.randint(4, 12)
        token = "".join(rng.choices(ALPHABET, k=length))
        vocab.append(f"{token}_{idx:05d}")
    return vocab


def _resolve_workers(requested: int | None, items: int) -> int:
    if requested is None or requested <= 0:
        requested = cpu_count() or 1
    requested = min(requested, items) if items else 1
    return max(1, requested)


def _chunk_ranges(total: int, chunks: int) -> Iterable[tuple[int, int]]:
    if total == 0:
        return []
    chunk_size = max(1, total // (chunks * 4 or 1))
    chunk_size = min(chunk_size, total)
    ranges = []
    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        ranges.append((start, end))
        start = end
    return ranges


def generate_zipf_weights(size: int, exponent: float) -> list[float]:
    if size <= 0:
        raise ValueError("Vocabulary size must be positive for Zipf weights")
    if exponent <= 0:
        raise ValueError("Zipf exponent must be positive")
    weights = [1.0 / ((idx + 1) ** exponent) for idx in range(size)]
    total = sum(weights)
    return [w / total for w in weights]


def _doc_chunk(
    start: int,
    end: int,
    words_per_doc: int,
    seed: int,
    vocabulary: Sequence[str],
    weights: Sequence[float],
) -> tuple[list[str], list[str], list[int]]:
    if not vocabulary:
        raise ValueError("Vocabulary must contain at least one token")
    if not weights:
        raise ValueError("Zipf weights are required for document generation")
    texts: list[str] = []
    names: list[str] = []
    pages: list[int] = []
    for idx in range(start, end):
        rng = random.Random(seed + idx)
        words = rng.choices(vocabulary, weights=weights, k=words_per_doc)
        texts.append(" ".join(words))
        names.append(f"lorem_doc_{idx:05d}.pdf")
        pages.append(rng.randint(1, 40))
    return texts, names, pages


def _query_chunk(
    start: int,
    end: int,
    terms_per_query: int,
    seed: int,
    vocabulary: Sequence[str],
) -> list[str]:
    if not vocabulary:
        raise ValueError("Vocabulary must contain at least one token")
    queries: list[str] = []
    limit = min(terms_per_query, len(vocabulary))
    for idx in range(start, end):
        rng = random.Random(seed + idx)
        terms = [] if limit == 0 else rng.sample(vocabulary, k=limit)
        queries.append(" ".join(terms))
    return queries


@dataclass
class BenchmarkResult:
    name: str
    documents: int
    queries: int
    add_seconds: float
    ingest_docs_per_sec: float
    query_seconds: float
    queries_per_sec: float
    avg_query_latency_ms: float

    def as_row(self) -> tuple[str, int, float, float, float, float]:
        return (
            self.name,
            self.documents,
            self.add_seconds,
            self.ingest_docs_per_sec,
            self.queries_per_sec,
            self.avg_query_latency_ms,
        )


def generate_documents(
    num_docs: int,
    words_per_doc: int,
    seed: int,
    vocabulary: Sequence[str],
    weights: Sequence[float],
    processes: int | None = None,
) -> tuple[list[str], list[str], list[int]]:
    workers = _resolve_workers(processes, num_docs)
    ranges = list(_chunk_ranges(num_docs, workers))
    if workers == 1 or len(ranges) == 1:
        chunks = [
            _doc_chunk(start, end, words_per_doc, seed, vocabulary, weights)
            for start, end in ranges
        ]
    else:
        with Pool(processes=workers) as pool:
            chunks = pool.starmap(
                _doc_chunk,
                [
                    (start, end, words_per_doc, seed, vocabulary, weights)
                    for start, end in ranges
                ],
            )
    texts: list[str] = []
    names: list[str] = []
    pages: list[int] = []
    for chunk_texts, chunk_names, chunk_pages in chunks:
        texts.extend(chunk_texts)
        names.extend(chunk_names)
        pages.extend(chunk_pages)
    return texts, names, pages


def generate_queries(
    num_queries: int,
    terms_per_query: int,
    seed: int,
    vocabulary: Sequence[str],
    processes: int | None = None,
) -> list[str]:
    workers = _resolve_workers(processes, num_queries)
    ranges = list(_chunk_ranges(num_queries, workers))
    if workers == 1 or len(ranges) == 1:
        chunks = [
            _query_chunk(start, end, terms_per_query, seed, vocabulary)
            for start, end in ranges
        ]
    else:
        with Pool(processes=workers) as pool:
            chunks = pool.starmap(
                _query_chunk,
                [
                    (start, end, terms_per_query, seed, vocabulary)
                    for start, end in ranges
                ],
            )
    queries: list[str] = []
    for chunk in chunks:
        queries.extend(chunk)
    return queries


def benchmark_index(
    name: str,
    index_cls: type[AbstractKeywordIndex],
    texts: Sequence[str],
    pdf_names: Sequence[str],
    pages: Sequence[int],
    queries: Sequence[str],
    k: int,
    index_root: Path,
) -> BenchmarkResult:
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
    ingest_docs_per_sec = (
        len(texts) / ingest_duration if ingest_duration else float("inf")
    )

    # Re-load the index to simulate real usage.
    index = index_cls(index_dir.as_posix())
    index.load_index()

    query_latencies: list[float] = []
    for query in queries:
        q_start = time.perf_counter()
        index.search(query, k)
        query_latencies.append(time.perf_counter() - q_start)
    total_query_time = sum(query_latencies)
    queries_per_sec = (
        len(queries) / total_query_time if total_query_time else float("inf")
    )
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
    )


def select_indexes(requested: Sequence[str]) -> dict[str, type[AbstractKeywordIndex]]:
    if not requested:
        return INDEX_REGISTRY
    selected: dict[str, type[AbstractKeywordIndex]] = {}
    for key in requested:
        lowered = key.lower()
        if lowered not in INDEX_REGISTRY:
            raise ValueError(
                "Unknown keyword index implementation "
                f"'{key}'. Options: {sorted(INDEX_REGISTRY)}"
            )
        selected[lowered] = INDEX_REGISTRY[lowered]
    return selected


def format_results(results: Iterable[BenchmarkResult]) -> str:
    header = (
        f"{'Index':<18} {'Docs':>8} {'Ingest (s)':>12} {'Docs/s':>12} "
        f"{'Queries/s':>12} {'Avg Latency (ms)':>18}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(
        f"{res.name:<18} {res.documents:>8} {res.add_seconds:>12.4f} "
        f"{res.ingest_docs_per_sec:>12.2f} {res.queries_per_sec:>12.2f} "
        f"{res.avg_query_latency_ms:>18.2f}"
        for res in results
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark keyword index implementations"
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=1000,
        help="Number of synthetic documents to generate",
    )
    parser.add_argument(
        "--words-per-document",
        type=int,
        default=100,
        help="Words per synthetic document",
    )
    parser.add_argument(
        "--queries", type=int, default=200, help="Number of random queries to execute"
    )
    parser.add_argument("--query-terms", type=int, default=3, help="Terms per query")
    parser.add_argument(
        "--k", type=int, default=5, help="Number of results to request per query"
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path("./.keyword_index_benchmark"),
        help="Directory where benchmark indices will be stored",
    )
    parser.add_argument(
        "--indexes",
        nargs="*",
        default=None,
        help=f"Subset of indexes to run ({', '.join(sorted(INDEX_REGISTRY))})",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=512,
        help="Size of the synthetic vocabulary used for docs/queries",
    )
    parser.add_argument(
        "--seed", type=int, default=13, help="PRNG seed for reproducibility"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help=(
            "Worker processes for generating documents and queries (default: cpu count)"
        ),
    )
    parser.add_argument(
        "--zipf-s",
        type=float,
        default=1.1,
        help="Zipf exponent used when sampling words for documents",
    )
    return parser.parse_args(argv)


def try_benchmark(
    name: str,
    index_cls: type[AbstractKeywordIndex],
    texts: Sequence[str],
    pdf_names: Sequence[str],
    pages: Sequence[int],
    queries: Sequence[str],
    k: int,
    index_root: Path,
) -> BenchmarkResult | None:
    try:
        return benchmark_index(
            name=name,
            index_cls=index_cls,
            texts=texts,
            pdf_names=pdf_names,
            pages=pages,
            queries=queries,
            k=k,
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

    vocab_seed = random.randrange(1 << 63)
    doc_seed = random.randrange(1 << 63)
    query_seed = random.randrange(1 << 63)
    vocabulary = generate_vocabulary(args.vocab_size, vocab_seed)
    zipf_weights = generate_zipf_weights(len(vocabulary), args.zipf_s)
    texts, pdf_names, pages = generate_documents(
        args.documents,
        args.words_per_document,
        seed=doc_seed,
        vocabulary=vocabulary,
        weights=zipf_weights,
        processes=args.processes,
    )
    queries = generate_queries(
        args.queries,
        args.query_terms,
        seed=query_seed,
        vocabulary=vocabulary,
        processes=args.processes,
    )

    selected = select_indexes(args.indexes or [])
    results: list[BenchmarkResult] = []
    for name, index_cls in selected.items():
        result = try_benchmark(
            name=name,
            index_cls=index_cls,
            texts=texts,
            pdf_names=pdf_names,
            pages=pages,
            queries=queries,
            k=args.k,
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
