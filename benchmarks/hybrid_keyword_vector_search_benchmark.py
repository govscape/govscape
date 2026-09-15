"""Benchmark hybrid search performance for parallel vs sequential execution.

This script builds a synthetic FAISS/vector index, a keyword index, and a
metadata index. It then compares timing for:
- existing textual hybrid search
- existing keyword hybrid search
- new combined keyword+vector hybrid search (sequential)
- new combined keyword+vector hybrid search (parallel)

Example:
    poetry run python benchmarks/hybrid_keyword_vector_search_benchmark.py \
        --documents 1000 --pages-per-doc 4 --queries 20 --k 20
"""

from __future__ import annotations

import argparse
import random
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from govscape.indexing import (
    FAISSIndex,
    SQLiteKeywordIndex,
    SQLiteMetadataIndex,
)
from govscape.indexing.hybrid import (
    HybridIndex,
    HybridKeywordMetadataIndex,
    HybridVectorMetadataIndex,
)
from govscape.query import EqualityPredicate, Predicate


@dataclass
class BenchmarkResult:
    name: str
    average_ms: float


def _parse_csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _build_documents(
    documents: int,
    pages_per_doc: int,
    dim: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    pdf_names = [f"doc_{idx:06d}.pdf" for idx in range(documents)]
    base_vectors = rng.normal(size=(documents, dim)).astype(np.float32)

    texts: list[str] = []
    digests: list[str] = []
    pages: list[str] = []
    vectors: list[np.ndarray] = []
    sub_domains: list[str] = []

    for idx, pdf_name in enumerate(pdf_names):
        sub_domain = "target.gov" if idx % 10 == 0 else "other.gov"
        sub_domains.append(sub_domain)

        for page in range(pages_per_doc):
            digests.append(pdf_name)
            pages.append(str(page))
            vectors.append(base_vectors[idx] + rng.normal(scale=0.05, size=(dim,)))
            texts.append(
                f"{pdf_name} page {page} \n{sub_domain} policy resilience data"
            )

    return pdf_names, digests, pages, texts, np.vstack(vectors), sub_domains


def _build_records(
    pdf_names: list[str], pages_per_doc: int, sub_domains: list[str]
) -> list[dict]:
    records: list[dict] = []
    for idx, pdf_name in enumerate(pdf_names):
        records.append(
            {
                "crawl_url": f"https://{sub_domains[idx]}/report/{pdf_name}",
                "crawl_date": "20240101",
                "digest": pdf_name,
                "pretty_name": pdf_name,
                "sub_domain": sub_domains[idx],
                "page_count": pages_per_doc,
            }
        )
    return records


def _prepare_metadata_index(
    index_dir: Path,
    records: list[dict],
    vector_store_key: str,
    vectors: np.ndarray,
    digests: list[str],
    pages: list[str],
) -> SQLiteMetadataIndex:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    metadata_index = SQLiteMetadataIndex(index_dir.as_posix())
    metadata_index.build_index()
    metadata_index.add_batch(records)
    metadata_index.upsert_vectors(vector_store_key, vectors, digests, pages)
    metadata_index.save_index()
    metadata_index.load_index()
    return metadata_index


def _prepare_keyword_index(
    index_dir: Path,
    texts: list[str],
    digests: list[str],
    pages: list[str],
) -> SQLiteKeywordIndex:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    keyword_index = SQLiteKeywordIndex(index_dir.as_posix())
    keyword_index.build_index()
    keyword_index.add_batch(texts, digests, pages)
    keyword_index.load_index()
    return keyword_index


def _prepare_faiss_index(
    index_dir: Path,
    vectors: np.ndarray,
    digests: list[str],
    pages: list[str],
) -> FAISSIndex:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_index = FAISSIndex(index_dir.as_posix())
    faiss_index.add_batch(vectors, digests, pages)
    return faiss_index


def _measure_search_time(
    search_fn,
    query_items,
) -> BenchmarkResult:
    times: list[float] = []
    for query in query_items:
        start = time.perf_counter()
        search_fn(*query)
        times.append(time.perf_counter() - start)
    return BenchmarkResult("", statistics.mean(times) * 1000)


def run_benchmark(
    documents: int,
    pages_per_doc: int,
    dim: int,
    queries: int,
    k: int,
    work_dir: Path,
    seed: int,
) -> list[BenchmarkResult]:
    pdf_names, digests, pages, texts, vectors, sub_domains = _build_documents(
        documents, pages_per_doc, dim, seed
    )
    records = _build_records(pdf_names, pages_per_doc, sub_domains)

    metadata_index = _prepare_metadata_index(
        work_dir / "metadata",
        records,
        "text",
        vectors,
        digests,
        pages,
    )
    keyword_index = _prepare_keyword_index(work_dir / "keyword", texts, digests, pages)
    faiss_index = _prepare_faiss_index(work_dir / "faiss", vectors, digests, pages)

    text_hybrid = HybridVectorMetadataIndex(
        vector_index=faiss_index,
        metadata_index=metadata_index,
        vector_store_key="text",
    )
    keyword_hybrid = HybridKeywordMetadataIndex(
        keyword_index=keyword_index,
        metadata_index=metadata_index,
    )
    combined_hybrid = HybridIndex([text_hybrid, keyword_hybrid])

    rng = random.Random(seed + 1)
    query_indices = [rng.randrange(len(vectors)) for _ in range(queries)]
    query_texts = [texts[idx] for idx in query_indices]
    query_vectors = [vectors[idx] for idx in query_indices]

    predicates: list[Predicate] = [EqualityPredicate("sub_domain", "target.gov")]

    results: list[BenchmarkResult] = []

    search_pairs = [
        (q_vec, q_txt) for q_vec, q_txt in zip(query_vectors, query_texts, strict=True)
    ]

    results.append(
        BenchmarkResult(
            name="textual_hybrid",
            average_ms=_measure_search_time(
                lambda q_vec, q_txt: text_hybrid.search(q_vec, predicates, k),
                search_pairs,
            ).average_ms,
        )
    )
    results.append(
        BenchmarkResult(
            name="keyword_hybrid",
            average_ms=_measure_search_time(
                lambda q_vec, q_txt: keyword_hybrid.search(q_txt, predicates, k),
                search_pairs,
            ).average_ms,
        )
    )
    results.append(
        BenchmarkResult(
            name="combined_hybrid_sequential",
            average_ms=_measure_search_time(
                lambda q_vec, q_txt: combined_hybrid.search(
                    [q_vec, q_txt], predicates, k, parallel=False
                ),
                search_pairs,
            ).average_ms,
        )
    )
    results.append(
        BenchmarkResult(
            name="combined_hybrid_parallel",
            average_ms=_measure_search_time(
                lambda q_vec, q_txt: combined_hybrid.search(
                    [q_vec, q_txt], predicates, k, parallel=True
                ),
                search_pairs,
            ).average_ms,
        )
    )

    return results


def format_results(rows: list[BenchmarkResult]) -> str:
    header = f"{'Search Mode':<28} {'Avg Latency (ms)':>18}"
    lines = [header, "-" * len(header)]
    lines.extend([f"{row.name:<28} {row.average_ms:>18.3f}" for row in rows])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark hybrid search performance."  # pragma: no cover
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=1000,
        help="Number of distinct PDF documents to generate.",
    )
    parser.add_argument(
        "--pages-per-doc",
        type=int,
        default=4,
        help="Number of pages per PDF document.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=384,
        help="Embedding dimensionality.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=20,
        help="Number of queries to benchmark.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Result count to request from each search.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/govscape_hybrid_search_bench"),
        help="Working directory for temporary index artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for synthetic data generation.",
    )
    args = parser.parse_args()

    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    rows = run_benchmark(
        documents=args.documents,
        pages_per_doc=args.pages_per_doc,
        dim=args.dim,
        queries=args.queries,
        k=args.k,
        work_dir=args.work_dir,
        seed=args.seed,
    )
    print(format_results(rows))


if __name__ == "__main__":
    main()
