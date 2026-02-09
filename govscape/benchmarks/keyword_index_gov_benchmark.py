"""Benchmark utilities for AbstractKeywordIndex implementations using real government documents.

This script loads real government documents from JSONL files, extracts their text content,
generates queries based on actual terms found in the documents, and benchmarks various
keyword index implementations for both ingestion and querying performance.

Example:
    poetry run python -m govscape.benchmarks.keyword_index_gov_benchmark \
        --queries 200 --data-dir ./govscape/benchmarks/docs
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Type

from govscape.indexing import (
    AbstractKeywordIndex,
    LanceDBKeywordIndex,
    SQLiteKeywordIndex,
    WhooshKeywordIndex
)

# Registry of available keyword index implementations
INDEX_REGISTRY: Dict[str, Type[AbstractKeywordIndex]] = {
    "lancedb": LanceDBKeywordIndex,
    "sqlite": SQLiteKeywordIndex,
    "whoosh": WhooshKeywordIndex,
    "lucene": LuceneKeywordIndex,
    "meilisearch": MeilisearchKeywordIndex,
}

try:  # pragma: no cover - optional dependency
    from govscape.indexing import ElasticsearchKeywordIndex  # type: ignore

    INDEX_REGISTRY["elasticsearch"] = ElasticsearchKeywordIndex
except Exception:  # pylint: disable=broad-except
    pass

try:  # pragma: no cover - optional dependency
    from govscape.indexing import LuceneKeywordIndex  # type: ignore

    INDEX_REGISTRY["lucene"] = LuceneKeywordIndex
except Exception:  # pylint: disable=broad-except
    pass

try:  # pragma: no cover - optional dependency
    from govscape.indexing import MeilisearchKeywordIndex  # type: ignore
    INDEX_REGISTRY["meilisearch"] = MeilisearchKeywordIndex
except Exception:  # pylint: disable=broad-except
    pass

def _resolve_workers(requested: int | None, items: int) -> int:
    """Resolve the number of worker processes to use."""
    if requested is None or requested <= 0:
        requested = cpu_count() or 1
    requested = min(requested, items) if items else 1
    return max(1, requested)


def _chunk_ranges(total: int, chunks: int) -> Iterable[Tuple[int, int]]:
    """Generate ranges for chunking work across processes."""
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


def load_documents_from_jsonl(data_dir: Path) -> List[Dict]:
    """Load documents from JSONL files in the specified directory.
    
    Args:
        data_dir: Directory containing JSONL files with government documents
        
    Returns:
        List of document dictionaries with 'id', 'text', and other fields
    """
    documents = []
    # can be .jsonl or .jsonl.txt
    jsonl_files = sorted(data_dir.glob("*.jsonl*")) + sorted(data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {data_dir}")
    
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line.strip())
                    if 'text' in doc and doc['text']:
                        documents.append(doc)
                except json.JSONDecodeError:
                    continue
    
    return documents


def extract_vocabulary_from_documents(documents: Sequence[Dict], min_freq: int = 2) -> List[str]:
    """Extract a vocabulary of meaningful terms from documents.
    
    Args:
        documents: List of document dictionaries with 'text' field
        min_freq: Minimum frequency for a term to be included in vocabulary
        
    Returns:
        List of terms sorted by frequency (most common first)
    """
    word_counts: Counter = Counter()
    
    for doc in documents:
        text = doc.get('text', '')
        # Simple tokenization: split on whitespace and punctuation
        words = text.lower().split()
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                      'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they', 'them', 'their'}
        filtered_words = [
            w.strip('.,;:!?()[]{}"\'-').lower() 
            for w in words 
            if len(w) > 3 and w.lower() not in stop_words
        ]
        word_counts.update(filtered_words)
    
    # Filter by minimum frequency and return sorted by frequency
    vocabulary = [word for word, count in word_counts.items() if count >= min_freq]
    # Sort by frequency (descending)
    vocabulary.sort(key=lambda w: word_counts[w], reverse=True)
    
    return vocabulary


def prepare_documents(
    documents: Sequence[Dict],
    seed: int
) -> Tuple[List[str], List[str], List[int]]:
    """Prepare documents for indexing by extracting text and generating metadata.
    
    Args:
        documents: List of document dictionaries
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (texts, pdf_names, page_numbers)
    """
    rng = random.Random(seed)
    texts = []
    pdf_names = []
    pages = []
    
    for idx, doc in enumerate(documents):
        text = doc.get('text', '')
        # Use doc ID if available, otherwise generate a name
        doc_id = doc.get('id', f'doc_{idx:05d}')
        pdf_name = f"gov_doc_{doc_id[:16]}.pdf"
        page_num = rng.randint(1, 50)  # Synthetic page numbers as requested
        
        texts.append(text)
        pdf_names.append(pdf_name)
        pages.append(page_num)
    
    return texts, pdf_names, pages


def _query_chunk(
    start: int,
    end: int,
    terms_per_query: int,
    seed: int,
    vocabulary: Sequence[str],
) -> List[str]:
    """Generate a chunk of queries from vocabulary."""
    if not vocabulary:
        raise ValueError("Vocabulary must contain at least one token")
    queries: List[str] = []
    limit = min(terms_per_query, len(vocabulary))
    for idx in range(start, end):
        rng = random.Random(seed + idx)
        if limit == 0:
            terms = []
        else:
            terms = rng.sample(vocabulary, k=limit)
        queries.append(" ".join(terms))
    return queries


def generate_queries(
    num_queries: int,
    terms_per_query: int,
    seed: int,
    vocabulary: Sequence[str],
    processes: int | None = None,
) -> List[str]:
    """Generate random queries from the vocabulary extracted from documents.
    
    Args:
        num_queries: Number of queries to generate
        terms_per_query: Number of terms per query
        seed: Random seed
        vocabulary: List of terms to sample from
        processes: Number of worker processes
        
    Returns:
        List of query strings
    """
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
    queries: List[str] = []
    for chunk in chunks:
        queries.extend(chunk)
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

    def as_row(self) -> Tuple[str, int, float, float, float, float]:
        return (
            self.name,
            self.documents,
            self.add_seconds,
            self.ingest_docs_per_sec,
            self.queries_per_sec,
            self.avg_query_latency_ms,
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
        f"{'Queries/s':>12} {'Avg Latency (ms)':>18}"
    )
    lines = [header, "-" * len(header)]
    for res in results:
        lines.append(
            f"{res.name:<18} {res.documents:>8} {res.add_seconds:>12.4f} "
            f"{res.ingest_docs_per_sec:>12.2f} "
            f"{res.queries_per_sec:>12.2f} {res.avg_query_latency_ms:>18.2f}"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark keyword index implementations using real government documents"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./govscape/benchmarks/docs"),
        help="Directory containing JSONL files with government documents",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=200,
        help="Number of random queries to execute",
    )
    parser.add_argument(
        "--query-terms",
        type=int,
        default=3,
        help="Terms per query",
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
        "--min-term-freq",
        type=int,
        default=2,
        help="Minimum term frequency for vocabulary extraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PRNG seed for reproducibility",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Worker processes for parallel operations (default: cpu count)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Maximum vocabulary size to use for query generation",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the government document benchmark."""
    args = parse_args(argv or sys.argv[1:])
    random.seed(args.seed)
    
    index_root = args.index_root.resolve()
    index_root.mkdir(parents=True, exist_ok=True)
    
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        return 1
    
    # Load documents
    print(f"[INFO] Loading documents from {data_dir}...", file=sys.stderr)
    documents = load_documents_from_jsonl(data_dir)
    if not documents:
        print("[ERROR] No documents loaded", file=sys.stderr)
        return 1
    print(f"[INFO] Loaded {len(documents)} documents", file=sys.stderr)
    
    # Extract vocabulary from documents
    print("[INFO] Extracting vocabulary from documents...", file=sys.stderr)
    vocabulary = extract_vocabulary_from_documents(documents, min_freq=args.min_term_freq)
    # Limit vocabulary size for query generation
    vocabulary = vocabulary[:args.vocab_size]
    print(f"[INFO] Extracted vocabulary of {len(vocabulary)} terms", file=sys.stderr)
    
    if not vocabulary:
        print("[ERROR] No vocabulary extracted from documents", file=sys.stderr)
        return 1
    
    # Prepare documents for indexing
    doc_seed = random.randrange(1 << 63)
    query_seed = random.randrange(1 << 63)
    texts, pdf_names, pages = prepare_documents(documents, seed=doc_seed)
    
    # Generate queries
    print(f"[INFO] Generating {args.queries} queries...", file=sys.stderr)
    queries = generate_queries(
        args.queries,
        args.query_terms,
        seed=query_seed,
        vocabulary=vocabulary,
        processes=args.processes,
    )
    
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
