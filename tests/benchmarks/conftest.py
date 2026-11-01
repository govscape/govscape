"""Shared fixtures for the CodSpeed PDF-processing benchmark suite.

These benchmarks use the full real-world PDF sample under `tests/test_data/`
(100 government PDFs, ~1,660 pages, ~96MB), rather than a small synthetic
sample, so measurements reflect realistic documents. That corpus is paired
with CDX crawl metadata (see `CONTRIBUTOR.md` -> "Running Govscape Locally"
for how it was generated): `tests/test_data/cdx/digests_manifest.csv` lists
each PDF's filename, digest, and source URL, and is used here as the
authoritative file list.

Because `pytest-codspeed` repeatedly invokes each benchmarked callable, and
this corpus is large, individual CI runs can take several minutes (see the
docstring in `test_pdf_processing_benchmarks.py` for details per benchmark).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

import numpy as np

import pandas as pd

from govscape.config import DataModel
from govscape.indexing.keyword import AbstractKeywordIndex
from govscape.indexing.metadata import AbstractMetadataIndex
from govscape.indexing.vector import AbstractVectorIndex
from govscape.utils import extract_subdomain

from pathlib import Path

import pytest

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
CDX_DIR = TEST_DATA_DIR / "cdx"
PDF_DIR = TEST_DATA_DIR / "pdfs"

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _timed_stage(name: str, fn: Callable[[], T]) -> T:
    """Run `fn`, logging a start/finish line with elapsed time (pair with
    `--log-cli-level=INFO` to see progress live)."""
    logger.info("Starting %s...", name)
    start = time.perf_counter()
    result = fn()
    logger.info("Finished %s in %.1fs", name, time.perf_counter() - start)
    return result


@pytest.fixture(scope="module")
def sample_pdf_files() -> list[str]:
    """The full real-world PDF sample, as listed in the CDX digest manifest."""
    with open(CDX_DIR / "digests_manifest.csv", newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    return sorted(str(PDF_DIR / row["file"]) for row in manifest_rows)


def read_cdx_metadata_records(data_model: DataModel) -> list[dict]:
    """Build metadata-index records by joining each PDF's extraction metadata
    with its matching CDX crawl record(s). A digest crawled more than once
    yields one record per crawl.
    """
    cdx = pd.read_parquet(CDX_DIR / "complete_cdx_sample.parquet")
    cdx["digest"] = cdx["digest"].str.replace("sha1:", "", regex=False)

    records: list[dict] = []
    if not os.path.isdir(data_model.metadata_directory):
        return records

    for digest_dir in sorted(
        os.scandir(data_model.metadata_directory), key=lambda e: e.name
    ):
        if not digest_dir.is_dir():
            continue
        digest = digest_dir.name
        metadata_path = data_model.metadata_file_path(digest)
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, encoding="utf-8") as f:
            pdf_metadata = json.load(f)

        for _, cdx_row in cdx[cdx["digest"] == digest].iterrows():
            records.append(
                {
                    "crawl_url": cdx_row["url"],
                    "crawl_date": cdx_row["crawl_date"],
                    "digest": digest,
                    "pretty_name": pdf_metadata.get("pretty_name", ""),
                    "sub_domain": extract_subdomain(cdx_row["url"]),
                    "page_count": pdf_metadata.get("num_pages", 0),
                }
            )
    return records


def _iter_npy_embeddings(embedding_dir: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Read every `.npy` embedding under `embedding_dir` (one subdirectory per
    digest, files named `{digest}_{page}.npy`), returning (embeddings,
    digests, pages).
    """
    embeddings: list[np.ndarray] = []
    digests: list[str] = []
    pages: list[str] = []
    if not os.path.isdir(embedding_dir):
        return np.empty((0, 0), dtype=np.float32), digests, pages

    for digest_dir in sorted(os.scandir(embedding_dir), key=lambda e: e.name):
        if not digest_dir.is_dir():
            continue
        digest = digest_dir.name
        for npy_file in sorted(os.listdir(digest_dir.path)):
            if not npy_file.endswith(".npy"):
                continue
            page = npy_file[: -len(".npy")].rpartition("_")[2]
            embeddings.append(np.load(os.path.join(digest_dir.path, npy_file)))
            digests.append(digest)
            pages.append(page)

    array = np.asarray(embeddings, dtype=np.float32) if embeddings else np.empty((0, 0))
    return array, digests, pages


def _iter_extracted_text(txt_dir: str) -> tuple[list[str], list[str], list[str]]:
    """Read every extracted `.txt` page file under `txt_dir`, returning
    (texts, digests, pages)."""
    texts: list[str] = []
    digests: list[str] = []
    pages: list[str] = []
    if not os.path.isdir(txt_dir):
        return texts, digests, pages

    for digest_dir in sorted(os.scandir(txt_dir), key=lambda e: e.name):
        if not digest_dir.is_dir():
            continue
        digest = digest_dir.name
        for txt_file in sorted(os.listdir(digest_dir.path)):
            if not txt_file.endswith(".txt"):
                continue
            page = txt_file[: -len(".txt")].rpartition("_")[2]
            with open(os.path.join(digest_dir.path, txt_file), encoding="utf-8") as f:
                texts.append(f.read())
            digests.append(digest)
            pages.append(page)
    return texts, digests, pages


def build_vector_index(
    vector_index_cls: type[AbstractVectorIndex], embedding_dir: str, index_dir: str
) -> tuple[AbstractVectorIndex, np.ndarray, list[str], list[str]]:
    """Build and save a vector index from `.npy` embeddings under
    `embedding_dir`. Also returns (embeddings, digests, pages), since a
    metadata index needs these too (via `upsert_vectors`).
    """
    embeddings, digests, pages = _iter_npy_embeddings(embedding_dir)
    index = vector_index_cls(index_dir)
    index.build_index()
    if len(embeddings):
        index.add_batch(embeddings, digests, pages)
    index.save_index()
    return index, embeddings, digests, pages


def build_keyword_index(
    keyword_index_cls: type[AbstractKeywordIndex], txt_dir: str, index_dir: str
) -> AbstractKeywordIndex:
    """Build and save a keyword index from extracted `.txt` page files under
    `txt_dir`."""
    texts, digests, pages = _iter_extracted_text(txt_dir)
    index = keyword_index_cls(index_dir)
    index.build_index()
    if texts:
        index.add_batch(texts, digests, pages)
    index.save_index()
    return index


def build_metadata_index(
    metadata_index_cls: type[AbstractMetadataIndex],
    index_dir: str,
    metadata_records: list[dict],
    vectors_by_store_key: dict[str, tuple[np.ndarray, list[str], list[str]]]
    | None = None,
) -> AbstractMetadataIndex:
    """Build and save a metadata index from `metadata_records`, optionally
    upserting per-digest/page vectors (needed for prefiltered hybrid search)
    under one or more `vector_store_key`s (e.g. "text", "visual")."""
    index = metadata_index_cls(index_dir)
    index.build_index()
    if metadata_records:
        index.add_batch(metadata_records)
    for vector_store_key, (vectors, digests, pages) in (
        vectors_by_store_key or {}
    ).items():
        if len(vectors):
            index.upsert_vectors(vector_store_key, vectors, digests, pages)
    index.save_index()
    return index


def write_total_pdfs_stat(data_model: DataModel, pdf_files: list[str]) -> None:
    """Write `data_model.stats_file`, read by `Server` for pagination."""
    digests = {Path(pdf_file).stem for pdf_file in pdf_files}
    with open(data_model.stats_file, "w", encoding="utf-8") as f:
        f.write(str(len(digests)))
