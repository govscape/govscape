"""Output-regression tests for the full index-build -> Server-load -> search
pipeline.

Complements `tests/test_pdf_processing_regressions.py` (extraction/embedding
output): this module covers the next stage -- building the four index types
`Server` loads (text + visual FAISS vector indices, a keyword index, and a
metadata index) from real extracted text and real CDX crawl data, loading a
real `Server` from them, and snapshotting real search results.

Unlike `tests/benchmarks/test_index_and_server_benchmarks.py` (which uses
`Dummy_TextEmbeddingModel`/`Dummy_VisualEmbeddingModel` for speed), this
module uses a small deterministic hash-based embedding model defined below:
the `Dummy_*EmbeddingModel` production classes draw from an *unseeded* RNG
and are non-deterministic across calls/runs, which would make vector-search
results unreproducible and defeat the purpose of a regression snapshot. The
hash-based model instead derives each embedding as a pure, reproducible
function of its input text, so the same build always produces the same
vectors and the same search rankings.

Keyword/textual search queries are derived from whatever text the *current*
run actually extracts from a fixed sample PDF (rather than a hardcoded
literal string), so this module only regression-tests the index/search
pipeline itself; extraction-wording drift is already covered independently by
`tests/test_pdf_processing_regressions.py`.

If a change intentionally alters these results, regenerate the baselines with
`poetry run pytest tests/test_index_and_server_regressions.py --force-regen`
and commit the updated files under
`tests/test_index_and_server_regressions/`.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import numpy as np

from govscape.config import DataModel, ServerConfig
from govscape.indexing import FAISSIndex, SQLiteKeywordIndex, SQLiteMetadataIndex
from govscape.processing import PDFExtractionStage
from govscape.query import EqualityPredicate, Query
from govscape.server import Server
from tests.benchmarks.conftest import (
    build_keyword_index,
    build_metadata_index,
    build_vector_index,
    read_cdx_metadata_records,
    sample_pdf_files,  # noqa: F401 (re-exported as a pytest fixture)
    write_total_pdfs_stat,
)

CPU_COUNT = os.cpu_count() or 1
EMBEDDING_DIM = 16


def _hash_embedding(text: str, dim: int) -> np.ndarray:
    """A pure, reproducible embedding function: the same input text always
    yields the same vector, regardless of run order or global RNG state
    (unlike `Dummy_TextEmbeddingModel`/`Dummy_VisualEmbeddingModel`, which
    draw from an unseeded RNG and are non-deterministic across calls)."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.random(dim).astype(np.float32)


class _DeterministicTextModel:
    """Minimal `text_model` stand-in for `ServerConfig`/`Server`, deterministic
    unlike the real `Dummy_TextEmbeddingModel`."""

    d = EMBEDDING_DIM

    def encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
        return _hash_embedding(text, self.d)


class _DeterministicVisualModel:
    """Minimal `visual_model` stand-in for `ServerConfig`/`Server`,
    deterministic unlike the real `Dummy_VisualEmbeddingModel`."""

    d = EMBEDDING_DIM

    def encode_text(self, text: str) -> np.ndarray:
        return _hash_embedding(text, self.d)


def _write_deterministic_embeddings(data_model: DataModel, dim: int) -> None:
    """Write deterministic `.npy` embeddings into `embedding_directory`
    (derived from each page's real extracted text) and
    `embedding_img_pg_directory` (derived from each page's digest/page
    identity, since there is no real image-content model here), using the
    same file layout `TextEmbeddingStage`/`PageImageEmbeddingStage` produce,
    so `build_vector_index` (from `tests/benchmarks/conftest.py`) can read
    them back unmodified.
    """
    for digest_dir in sorted(
        os.scandir(data_model.txt_directory), key=lambda e: e.name
    ):
        if not digest_dir.is_dir():
            continue
        digest = digest_dir.name
        out_dir = data_model.embedding_pdf_directory(digest)
        os.makedirs(out_dir, exist_ok=True)
        for txt_file in sorted(os.listdir(digest_dir.path)):
            if not txt_file.endswith(".txt"):
                continue
            page = txt_file[: -len(".txt")].rpartition("_")[2]
            with open(os.path.join(digest_dir.path, txt_file), encoding="utf-8") as f:
                text = f.read()
            np.save(
                os.path.join(out_dir, f"{digest}_{page}.npy"),
                _hash_embedding(text, dim),
            )

    for digest_dir in sorted(
        os.scandir(data_model.image_directory), key=lambda e: e.name
    ):
        if not digest_dir.is_dir():
            continue
        digest = digest_dir.name
        out_dir = data_model.embedding_img_pg_pdf_directory(digest)
        os.makedirs(out_dir, exist_ok=True)
        for img_file in sorted(os.listdir(digest_dir.path)):
            if not img_file.endswith(".jpeg"):
                continue
            page = img_file[: -len(".jpeg")].rpartition("_")[2]
            np.save(
                os.path.join(out_dir, f"{digest}_{page}.npy"),
                _hash_embedding(f"{digest}_{page}", dim),
            )


def _first_nonempty_page_text(data_model: DataModel, digest: str) -> tuple[str, str]:
    """Return (page_number, full_text) for `digest`'s first non-empty
    extracted page."""
    txt_dir = Path(data_model.txt_pdf_directory(digest))
    for txt_file in sorted(txt_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        if text.strip():
            page = txt_file.stem.rpartition("_")[2]
            return page, text
    raise AssertionError(f"No non-empty extracted page found for digest {digest}")


def _sanitize_result(result: dict) -> dict:
    """Project a Server search result to deterministic, regression-safe
    fields: `jpeg` is an absolute tmp-dir path (changes every run) so only
    its basename is kept, and `distance` is rounded to avoid float-precision
    noise."""
    sanitized = dict(result)
    sanitized["jpeg"] = Path(result["jpeg"]).name
    sanitized["distance"] = round(result["distance"], 4)
    return sanitized


@pytest.fixture(scope="module")
def extracted_data_model(
    sample_pdf_files,  # noqa: F811
    tmp_path_factory,
) -> DataModel:
    """Run real PDF extraction (text, images, per-PDF metadata) once."""
    data_dir = tmp_path_factory.mktemp("index_server_regressions")
    data_model = DataModel(data_dir.as_posix())
    PDFExtractionStage(
        data_model=data_model, pdf_files=sample_pdf_files, cpu_count=CPU_COUNT
    ).run()
    return data_model


@pytest.fixture(scope="module")
def server_data_model(
    extracted_data_model,
    sample_pdf_files,  # noqa: F811
) -> DataModel:
    """`extracted_data_model`, with all four indices built (using
    deterministic embeddings) directly into its own `index_*` directories."""
    _write_deterministic_embeddings(extracted_data_model, EMBEDDING_DIM)

    build_keyword_index(
        SQLiteKeywordIndex,
        extracted_data_model.txt_directory,
        extracted_data_model.index_keyword_directory,
    )

    text_index, text_vectors, text_digests, text_pages = build_vector_index(
        FAISSIndex,
        extracted_data_model.embedding_directory,
        extracted_data_model.index_directory,
    )
    visual_index, visual_vectors, visual_digests, visual_pages = build_vector_index(
        FAISSIndex,
        extracted_data_model.embedding_img_pg_directory,
        extracted_data_model.index_img_pg_directory,
    )

    metadata_records = read_cdx_metadata_records(extracted_data_model)
    build_metadata_index(
        SQLiteMetadataIndex,
        extracted_data_model.index_metadata_directory,
        metadata_records,
        vectors_by_store_key={
            "text": (text_vectors, text_digests, text_pages),
            "visual": (visual_vectors, visual_digests, visual_pages),
        },
    )

    write_total_pdfs_stat(extracted_data_model, sample_pdf_files)
    return extracted_data_model


@pytest.fixture(scope="module")
def server(server_data_model: DataModel) -> Server:
    config = ServerConfig(
        server_data_model.data_dir,
        _DeterministicTextModel(),
        _DeterministicVisualModel(),
        vector_index_type="Memory",
        keyword_index_type="SQLite",
        k=5,
    )
    return Server(config)


def test_index_sizes_regression(server: Server, data_regression):
    """Built-index entry counts should not change unexpectedly."""
    data_regression.check(
        {
            "keyword_index_entries": server.keyword_index.total_entries(),
            "text_vector_index_entries": server.text_index.total_entries(),
            "visual_vector_index_entries": server.visual_index.total_entries(),
            "metadata_index_entries": server.metadata_index.total_entries(),
        }
    )


def test_server_keyword_search_regression(
    server: Server,
    server_data_model: DataModel,
    sample_pdf_files,  # noqa: F811
    data_regression,
):
    """A keyword-phrase query built from a real document's real extracted
    text should keep surfacing that document."""
    target_digest = Path(sample_pdf_files[0]).stem
    _page, text = _first_nonempty_page_text(server_data_model, target_digest)
    phrase = " ".join(text.split()[:6])

    response = server.search(Query(f'"{phrase}"', search_type="keyword"))

    assert any(r["pdf"] == target_digest for r in response.results)
    data_regression.check(
        {
            "pagination": response.pagination,
            "results": [_sanitize_result(r) for r in response.results],
        }
    )


def test_server_textual_search_regression(
    server: Server,
    server_data_model: DataModel,
    sample_pdf_files,  # noqa: F811
    data_regression,
):
    """Querying with a page's own text makes the deterministic hash-based
    query embedding identical to that page's indexed embedding, so the top
    textual-search result is reproducibly that exact (digest, page)."""
    target_digest = Path(sample_pdf_files[0]).stem
    page, text = _first_nonempty_page_text(server_data_model, target_digest)

    response = server.search(Query(text, search_type="textual"))

    assert response.results[0]["pdf"] == target_digest
    assert response.results[0]["page"] == page
    data_regression.check(
        {
            "pagination": response.pagination,
            "results": [_sanitize_result(r) for r in response.results],
        }
    )


def test_server_textual_search_with_predicate_regression(
    server: Server,
    server_data_model: DataModel,
    sample_pdf_files,  # noqa: F811
    data_regression,
):
    """A metadata predicate should correctly filter vector search results,
    exercising the metadata-driven prefilter/postfilter strategy selection
    (see `govscape.indexing.hybrid`)."""
    target_digest = Path(sample_pdf_files[0]).stem
    _page, text = _first_nonempty_page_text(server_data_model, target_digest)

    metadata_records = read_cdx_metadata_records(server_data_model)
    target_sub_domain = next(
        r["sub_domain"]
        for r in metadata_records
        if r["digest"] == target_digest and r.get("sub_domain")
    )

    response = server.search(
        Query(
            text,
            search_type="textual",
            predicates=[EqualityPredicate("sub_domain", target_sub_domain)],
        )
    )

    assert response.results[0]["pdf"] == target_digest
    for result in response.results:
        assert result["sub_domain"] == target_sub_domain
    data_regression.check(
        {
            "pagination": response.pagination,
            "results": [_sanitize_result(r) for r in response.results],
        }
    )
