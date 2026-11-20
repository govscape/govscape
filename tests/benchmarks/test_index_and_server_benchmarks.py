"""CodSpeed benchmarks for building search indices, loading them onto the
Server, and querying it.

Complements `test_pdf_processing_benchmarks.py` (extraction + embedding
compute cost) by covering the next pipeline stage. Uses the cheap `Dummy`
text/visual embedding models so these benchmarks measure index-build/
server-load/query performance in isolation from embedding inference cost.

Run locally with `poetry run pytest tests/benchmarks --codspeed`.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest

from govscape.config import DataModel, ServerConfig
from govscape.indexing import FAISSIndex, SQLiteKeywordIndex, SQLiteMetadataIndex
from govscape.processing import (
    PageImageEmbeddingStage,
    PDFExtractionStage,
    TextEmbeddingStage,
)
from govscape.query import EqualityPredicate, Query
from govscape.server import Server
from govscape.text_embedding_models import Dummy_TextEmbeddingModel
from govscape.visual_embedding_models import Dummy_VisualEmbeddingModel

from .conftest import (
    _timed_stage,
    build_keyword_index,
    build_metadata_index,
    build_vector_index,
    read_cdx_metadata_records,
    write_total_pdfs_stat,
)

CPU_COUNT = os.cpu_count() or 1

# Guaranteed present in essentially any English-language extracted text.
KEYWORD_QUERY = "the"


@pytest.fixture(scope="module")
def processed_data_model(sample_pdf_files, tmp_path_factory) -> DataModel:
    """Run extraction + cheap Dummy embedding once, shared by every benchmark
    below."""
    data_dir = tmp_path_factory.mktemp("index_server_benchmarks_source")
    data_model = DataModel(data_dir.as_posix())
    PDFExtractionStage(
        data_model=data_model, pdf_files=sample_pdf_files, cpu_count=CPU_COUNT
    ).run()
    TextEmbeddingStage(data_model=data_model, model_type="Dummy").run()
    PageImageEmbeddingStage(
        data_model=data_model, model_type="Dummy", cpu_count=CPU_COUNT
    ).run()
    return data_model


def test_keyword_index_build_benchmark(benchmark, processed_data_model, tmp_path: Path):
    """Benchmark turning extracted text into a queryable keyword index."""

    def run():
        index_dir = tmp_path / uuid.uuid4().hex
        index = build_keyword_index(
            SQLiteKeywordIndex, processed_data_model.txt_directory, str(index_dir)
        )
        total = index.total_entries()
        shutil.rmtree(index_dir, ignore_errors=True)
        return total

    total = benchmark.pedantic(
        lambda: _timed_stage("keyword index build", run),
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
    assert total > 0


def test_metadata_index_build_benchmark(
    benchmark, processed_data_model, tmp_path: Path
):
    """Benchmark turning CDX crawl data + per-PDF metadata into a queryable
    metadata index."""
    metadata_records = read_cdx_metadata_records(processed_data_model)

    def run():
        index_dir = tmp_path / uuid.uuid4().hex
        index = build_metadata_index(
            SQLiteMetadataIndex, str(index_dir), metadata_records
        )
        total = index.total_entries()
        shutil.rmtree(index_dir, ignore_errors=True)
        return total

    total = benchmark.pedantic(
        lambda: _timed_stage("metadata index build", run),
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
    assert total == len(metadata_records)


def test_vector_indices_build_benchmark(
    benchmark, processed_data_model, tmp_path: Path
):
    """Benchmark turning text/visual embeddings into queryable FAISS vector
    indices."""

    def run():
        index_dir = tmp_path / uuid.uuid4().hex
        text_index, *_ = build_vector_index(
            FAISSIndex,
            processed_data_model.embedding_directory,
            str(index_dir / "text"),
        )
        visual_index, *_ = build_vector_index(
            FAISSIndex,
            processed_data_model.embedding_img_pg_directory,
            str(index_dir / "visual"),
        )
        totals = (text_index.total_entries(), visual_index.total_entries())
        shutil.rmtree(index_dir, ignore_errors=True)
        return totals

    text_total, visual_total = benchmark.pedantic(
        lambda: _timed_stage("text/visual vector index build", run),
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
    assert text_total > 0
    assert visual_total > 0


@pytest.fixture(scope="module")
def server_ready_data_model(processed_data_model, sample_pdf_files) -> DataModel:
    """`processed_data_model`, with all four indices built into its own
    `index_*` directories."""
    build_keyword_index(
        SQLiteKeywordIndex,
        processed_data_model.txt_directory,
        processed_data_model.index_keyword_directory,
    )

    text_index, text_vectors, text_digests, text_pages = build_vector_index(
        FAISSIndex,
        processed_data_model.embedding_directory,
        processed_data_model.index_directory,
    )
    visual_index, visual_vectors, visual_digests, visual_pages = build_vector_index(
        FAISSIndex,
        processed_data_model.embedding_img_pg_directory,
        processed_data_model.index_img_pg_directory,
    )

    metadata_records = read_cdx_metadata_records(processed_data_model)
    build_metadata_index(
        SQLiteMetadataIndex,
        processed_data_model.index_metadata_directory,
        metadata_records,
        vectors_by_store_key={
            "text": (text_vectors, text_digests, text_pages),
            "visual": (visual_vectors, visual_digests, visual_pages),
        },
    )

    write_total_pdfs_stat(processed_data_model, sample_pdf_files)
    return processed_data_model


def _build_server(data_model: DataModel) -> Server:
    config = ServerConfig(
        data_model.data_dir,
        Dummy_TextEmbeddingModel(),
        Dummy_VisualEmbeddingModel(),
        vector_index_type="Memory",
        keyword_index_type="SQLite",
    )
    return Server(config)


def test_server_load_benchmark(benchmark, server_ready_data_model: DataModel):
    """Benchmark constructing a `Server` from already-built indices on
    disk."""
    server = benchmark.pedantic(
        lambda: _timed_stage(
            "server load", lambda: _build_server(server_ready_data_model)
        ),
        rounds=1,
        warmup_rounds=0,
        iterations=1,
    )
    assert server.text_index.total_entries() > 0
    assert server.visual_index.total_entries() > 0
    assert server.keyword_index.total_entries() > 0


def test_server_keyword_query_benchmark(benchmark, server_ready_data_model: DataModel):
    """Benchmark a real keyword search against a fully-loaded Server."""
    server = _build_server(server_ready_data_model)

    def run():
        return server.search(Query(KEYWORD_QUERY, search_type="keyword"))

    response = benchmark(run)
    assert response.pagination["page"] == 1


def test_server_textual_query_benchmark(benchmark, server_ready_data_model: DataModel):
    """Benchmark a real textual (vector) search against a fully-loaded
    Server."""
    server = _build_server(server_ready_data_model)

    def run():
        return server.search(Query("government policy report", search_type="textual"))

    response = benchmark(run)
    assert response.pagination["page"] == 1


def test_server_hybrid_query_with_predicate_benchmark(
    benchmark, server_ready_data_model: DataModel
):
    """Benchmark a hybrid search with a metadata predicate (exercises the
    prefilter/postfilter strategy selection in `govscape.indexing.hybrid`)."""
    server = _build_server(server_ready_data_model)
    metadata_records = read_cdx_metadata_records(server_ready_data_model)
    sub_domain = next(r["sub_domain"] for r in metadata_records if r.get("sub_domain"))

    def run():
        return server.search(
            Query(
                "government policy report",
                search_type="hybrid",
                predicates=[EqualityPredicate("sub_domain", sub_domain)],
            )
        )

    response = benchmark(run)
    assert response.pagination["page"] == 1
