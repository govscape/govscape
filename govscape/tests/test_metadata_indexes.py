# AI modified: 2026-04-12 18:24:13 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 18:35:40 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-03-09 764fe895
# AI modified: 2026-03-14 21:55:15 1c688b19
# AI modified: 2026-04-06 00:10:53 434ce298
# AI modified: 2026-04-12 23:51:04 fddc6344a807a84c8b9161bd3ffeded5153c5e27
import pytest

import numpy as np

from govscape.indexing import (
    DuckDBMetadataIndex,
    SQLiteMetadataIndex,
)

_RECORDS = [
    {
        "crawl_url": "https://epa.gov/reports/air_quality.pdf",
        "crawl_date": "20220315",
        "pdf_name": "air_quality.pdf",
        "sub_domain": "epa.gov",
        "page_count": 42,
        "s3_url": "s3://govscape/epa.gov/air_quality.pdf",
    },
    {
        "crawl_url": "https://energy.gov/reports/solar_grid.pdf",
        "crawl_date": "20230601",
        "pdf_name": "solar_grid.pdf",
        "sub_domain": "energy.gov",
        "page_count": 18,
        "s3_url": "s3://govscape/energy.gov/solar_grid.pdf",
    },
    {
        "crawl_url": "https://epa.gov/reports/water_quality.pdf",
        "crawl_date": "20240101",
        "pdf_name": "water_quality.pdf",
        "sub_domain": "epa.gov",
        "page_count": 55,
        "s3_url": "s3://govscape/epa.gov/water_quality.pdf",
    },
    {
        "crawl_url": "https://nasa.gov/reports/climate_data.pdf",
        "crawl_date": "20210820",
        "pdf_name": "climate_data.pdf",
        "sub_domain": "nasa.gov",
        "page_count": 77,
        "s3_url": "s3://govscape/nasa.gov/climate_data.pdf",
    },
    # Same pdf_name crawled twice from the same domain on different dates.
    {
        "crawl_url": "https://epa.gov/reports/air_quality.pdf",
        "crawl_date": "20230101",
        "pdf_name": "air_quality.pdf",
        "sub_domain": "epa.gov",
        "page_count": 44,
        "s3_url": "s3://govscape/epa.gov/air_quality.pdf",
    },
]


@pytest.fixture(
    params=[SQLiteMetadataIndex, DuckDBMetadataIndex],
    ids=["sqlite", "duckdb"],
)
def index(request, tmp_path):
    index_cls = request.param
    idx = index_cls((tmp_path / "metadata_index").as_posix())
    idx.build_index()
    idx.add_batch(_RECORDS)
    idx.save_index()

    reloaded = index_cls((tmp_path / "metadata_index").as_posix())
    reloaded.load_index()
    return reloaded


def test_total_entries(index):
    assert index.total_entries() == len(_RECORDS)


def test_no_filter_returns_matching_records(index):
    result = index.search(["air_quality.pdf", "solar_grid.pdf"])
    assert set(result.keys()) == {"air_quality.pdf", "solar_grid.pdf"}
    # air_quality.pdf was crawled twice
    assert len(result["air_quality.pdf"]) == 2
    assert len(result["solar_grid.pdf"]) == 1


def test_unknown_pdf_name_not_in_result(index):
    result = index.search(["does_not_exist.pdf"])
    assert result == {}


def test_no_filter_result_shape(index):
    result = index.search(["solar_grid.pdf"])
    record = result["solar_grid.pdf"][0]
    assert record["pdf_name"] == "solar_grid.pdf"
    assert record["sub_domain"] == "energy.gov"
    assert record["page_count"] == 18
    # crawl_date must be normalised to YYYY-MM-DD
    assert record["crawl_date"] == "2023-06-01"


def test_domain_filter(index):
    all_names = [r["pdf_name"] for r in _RECORDS]
    result = index.search(all_names, {"sub_domain": "epa.gov"})
    for entries in result.values():
        for entry in entries:
            assert entry["sub_domain"] == "epa.gov"
    assert "solar_grid.pdf" not in result
    assert "climate_data.pdf" not in result


def test_date_filter_crawled_after(index):
    all_names = [r["pdf_name"] for r in _RECORDS]
    result = index.search(all_names, {"crawled_after": "2023-01-01"})
    for entries in result.values():
        for entry in entries:
            assert entry["crawl_date"] >= "2023-01-01"
    assert "climate_data.pdf" not in result


def test_date_filter_crawled_before(index):
    all_names = [r["pdf_name"] for r in _RECORDS]
    result = index.search(all_names, {"crawled_before": "2022-12-31"})
    for entries in result.values():
        for entry in entries:
            assert entry["crawl_date"] <= "2022-12-31"


def test_all_filters_combined(index):
    all_names = [r["pdf_name"] for r in _RECORDS]
    result = index.search(
        all_names,
        {
            "sub_domain": "epa.gov",
            "crawled_after": "2022-01-01",
            "crawled_before": "2023-12-31",
        },
    )
    for entries in result.values():
        for entry in entries:
            assert entry["sub_domain"] == "epa.gov"
            assert "2022-01-01" <= entry["crawl_date"] <= "2023-12-31"


def test_unknown_filter_keys_are_ignored_in_search(index):
    all_names = [r["pdf_name"] for r in _RECORDS]
    baseline = index.search(all_names, {"sub_domain": "epa.gov"})
    with_unknown = index.search(
        all_names,
        {
            "sub_domain": "epa.gov",
            "unknown_key": "ignored-value",
        },
    )

    assert with_unknown == baseline


def test_filter_no_matches_returns_empty(index):
    result = index.search(["air_quality.pdf"], {"sub_domain": "nasa.gov"})
    assert result == {}


def test_count_filtered_documents(index):
    assert index.count_filtered_documents({"sub_domain": "epa.gov"}) == 2


def test_unknown_filter_keys_are_ignored_in_document_counts(index):
    baseline = index.count_filtered_documents({"sub_domain": "epa.gov"})
    with_unknown = index.count_filtered_documents(
        {
            "sub_domain": "epa.gov",
            "unknown_key": "ignored-value",
        }
    )
    assert with_unknown == baseline


def test_get_filtered_pdf_page_counts(index):
    result = index.get_filtered_pdf_page_counts({"sub_domain": "epa.gov"})
    # air_quality.pdf appears twice, so max(page_count)=44 is expected.
    assert result["air_quality.pdf"] == 44
    assert result["water_quality.pdf"] == 55


def test_vector_store_roundtrip(index):
    names = ["air_quality.pdf", "solar_grid.pdf"]
    pages = ["0", "1"]
    vectors = [
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
    ]

    index.upsert_vectors_batch("textual", names, pages, vectors)
    got = index.get_vectors_for_pdf_page_counts(
        "textual",
        {
            "air_quality.pdf": 1,
            "solar_grid.pdf": 2,
        },
    )

    assert "air_quality.pdf" in got
    assert "solar_grid.pdf" in got
    got_air = got["air_quality.pdf"][0][1]
    got_solar = got["solar_grid.pdf"][0][1]
    assert np.allclose(got_air, vectors[0])
    assert np.allclose(got_solar, vectors[1])
