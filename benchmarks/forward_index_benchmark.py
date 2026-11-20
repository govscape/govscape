"""Benchmark forward index implementations (SQLite vs LMDB).

Builds each forward index from synthetic page vectors in batches, like
generate_index_embedding does, then times get_vectors_for_digests for random
candidate sets of several sizes, which is what a prefiltered search does.
Lookups run on a warm OS page cache.

Example:
    poetry run python -m benchmarks.forward_index_benchmark \
        --num-docs 200000 --pages-per-doc 5 --dim 512 \
        --lookup-sizes 1 100 1000 10000 50000
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from govscape.indexing import FORWARD_INDEX_TYPES, build_forward_index


@dataclass
class BuildResult:
    forward_index_type: str
    vectors: int
    build_seconds: float
    disk_mib: float


@dataclass
class LookupResult:
    forward_index_type: str
    num_digests: int
    median_ms: float
    vectors_per_sec: float


def _digests(num_docs: int) -> list[str]:
    return [f"{i:032d}" for i in range(num_docs)]


def _dir_size_mib(path: Path) -> float:
    total = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, files in os.walk(path)
        for name in files
    )
    return total / (1024 * 1024)


def build(
    forward_index_type: str,
    index_dir: Path,
    digests: list[str],
    pages_per_doc: int,
    dim: int,
    batch_docs: int,
    seed: int,
) -> BuildResult:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    rng = np.random.default_rng(seed)
    index = build_forward_index(forward_index_type, str(index_dir))
    index.build_index()

    start = time.perf_counter()
    for i in range(0, len(digests), batch_docs):
        batch = digests[i : i + batch_docs]
        vectors = rng.random((len(batch) * pages_per_doc, dim), dtype=np.float32)
        batch_digests = [d for d in batch for _ in range(pages_per_doc)]
        batch_pages = [str(p) for _ in batch for p in range(pages_per_doc)]
        index.add_batch(vectors, batch_digests, batch_pages)
    index.save_index()
    build_seconds = time.perf_counter() - start
    index.close()

    return BuildResult(
        forward_index_type=forward_index_type,
        vectors=len(digests) * pages_per_doc,
        build_seconds=build_seconds,
        disk_mib=_dir_size_mib(index_dir),
    )


def time_lookups(
    forward_index_type: str,
    index_dir: Path,
    digests: list[str],
    lookup_sizes: Sequence[int],
    repeats: int,
    seed: int,
) -> list[LookupResult]:
    rng = random.Random(seed)
    index = build_forward_index(forward_index_type, str(index_dir))
    index.load_index()
    # Warm the page cache and open the handle outside the timed runs.
    index.get_vectors_for_digests(set(digests))

    results = []
    for size in lookup_sizes:
        timings = []
        num_vectors = 0
        for _ in range(repeats):
            candidates = set(rng.sample(digests, min(size, len(digests))))
            start = time.perf_counter()
            vectors, _, _ = index.get_vectors_for_digests(candidates)
            timings.append(time.perf_counter() - start)
            num_vectors = vectors.shape[0]
        median = statistics.median(timings)
        results.append(
            LookupResult(
                forward_index_type=forward_index_type,
                num_digests=size,
                median_ms=median * 1000,
                vectors_per_sec=num_vectors / median if median > 0 else 0.0,
            )
        )
    index.close()
    return results


def format_results(builds: list[BuildResult], lookups: list[LookupResult]) -> str:
    lines = [
        f"{'Type':<8} {'Vectors':>10} {'Build s':>9} {'Disk MiB':>10}",
        "-" * 40,
    ]
    lines.extend(
        f"{b.forward_index_type:<8} {b.vectors:>10} {b.build_seconds:>9.1f} "
        f"{b.disk_mib:>10.0f}"
        for b in builds
    )
    lines += [
        "",
        (
            f"{'Type':<8} {'Digests':>8} {'Median ms':>10} {'Vectors/s':>12} "
            f"{'vs SQLite':>10}"
        ),
        "-" * 52,
    ]
    sqlite_ms = {
        r.num_digests: r.median_ms for r in lookups if r.forward_index_type == "SQLite"
    }
    for r in lookups:
        base = sqlite_ms.get(r.num_digests)
        speedup = f"{base / r.median_ms:.2f}x" if base and r.median_ms else ""
        lines.append(
            f"{r.forward_index_type:<8} {r.num_digests:>8} {r.median_ms:>10.2f} "
            f"{r.vectors_per_sec:>12.0f} {speedup:>10}"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--types", nargs="+", default=FORWARD_INDEX_TYPES)
    parser.add_argument("--num-docs", type=int, default=200_000)
    parser.add_argument("--pages-per-doc", type=int, default=5)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--batch-docs", type=int, default=10_000)
    parser.add_argument(
        "--lookup-sizes", type=int, nargs="+", default=[1, 100, 1000, 10000, 50000]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--work-dir", type=Path, default=Path(".forward_index_bench_work")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    digests = _digests(args.num_docs)
    builds = []
    lookups = []
    for forward_index_type in args.types:
        index_dir = args.work_dir / forward_index_type
        print(f"== {forward_index_type}: building")
        builds.append(
            build(
                forward_index_type,
                index_dir,
                digests,
                args.pages_per_doc,
                args.dim,
                args.batch_docs,
                args.seed,
            )
        )
        print(f"== {forward_index_type}: timing lookups")
        lookups.extend(
            time_lookups(
                forward_index_type,
                index_dir,
                digests,
                args.lookup_sizes,
                args.repeats,
                args.seed,
            )
        )
    print()
    print(format_results(builds, lookups))


if __name__ == "__main__":
    main()
