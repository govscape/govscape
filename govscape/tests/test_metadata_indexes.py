# AI modified: 20260309 764fe895
# AI modified: 2026-04-26T22:00:43Z eac4f332
import pytest

from govscape.indexing import (
    DuckDBMetadataIndex,
    SQLiteMetadataIndex,
)
from govscape.query import EqualityPredicate, RangePredicate

_RECORDS = [
    {
        "crawl_url": "https://epa.gov/reports/air_quality.pdf",
        "crawl_date": "20220315",
        "digest": "air_quality.pdf",
        "pretty_name": "Air Quality Report",
        "sub_domain": "epa.gov",
        "page_count": 42,
    },
    {
        "crawl_url": "https://energy.gov/reports/solar_grid.pdf",
        "crawl_date": "20230601",
        "digest": "solar_grid.pdf",
        "pretty_name": "Solar Grid Analysis",
        "sub_domain": "energy.gov",
        "page_count": 18,
    },
    {
        "crawl_url": "https://epa.gov/reports/water_quality.pdf",
        "crawl_date": "20240101",
        "digest": "water_quality.pdf",
        "pretty_name": "Water Quality Study",
        "sub_domain": "epa.gov",
        "page_count": 55,
    },
    {
        "crawl_url": "https://nasa.gov/reports/climate_data.pdf",
        "crawl_date": "20210820",
        "digest": "climate_data.pdf",
        "pretty_name": "Climate Data Summary",
        "sub_domain": "nasa.gov",
        "page_count": 77,
    },
    # Same digest crawled twice from the same domain on different dates.
    {
        "crawl_url": "https://epa.gov/reports/air_quality.pdf",
        "crawl_date": "20230101",
        "digest": "air_quality.pdf",
        "pretty_name": "Air Quality Report",
        "sub_domain": "epa.gov",
        "page_count": 44,
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


def test_no_predicate_returns_matching_records(index):
    result = index.search(["air_quality.pdf", "solar_grid.pdf"])
    assert set(result.keys()) == {"air_quality.pdf", "solar_grid.pdf"}
    # air_quality.pdf was crawled twice
    assert len(result["air_quality.pdf"]) == 2
    assert len(result["solar_grid.pdf"]) == 1


def test_unknown_digest_not_in_result(index):
    result = index.search(["does_not_exist.pdf"])
    assert result == {}


def test_no_predicate_result_shape(index):
    result = index.search(["solar_grid.pdf"])
    record = result["solar_grid.pdf"][0]
    assert record["digest"] == "solar_grid.pdf"
    assert record["pretty_name"] == "Solar Grid Analysis"
    assert record["sub_domain"] == "energy.gov"
    assert record["page_count"] == 18
    # crawl_date must be normalised to YYYYMMDD
    assert record["crawl_date"] == "20230601"


def test_domain_predicate(index):
    all_names = [r["digest"] for r in _RECORDS]
    result = index.search(all_names, [EqualityPredicate("sub_domain", "epa.gov")])
    for entries in result.values():
        for entry in entries:
            assert entry["sub_domain"] == "epa.gov"
    assert "solar_grid.pdf" not in result
    assert "climate_data.pdf" not in result


def test_date_predicate_crawled_after(index):
    all_names = [r["digest"] for r in _RECORDS]
    result = index.search(all_names, [RangePredicate("crawl_date", min_val="20230101")])
    for entries in result.values():
        for entry in entries:
            assert entry["crawl_date"] >= "20230101"
    assert "climate_data.pdf" not in result


def test_date_predicate_crawled_before(index):
    all_names = [r["digest"] for r in _RECORDS]
    result = index.search(all_names, [RangePredicate("crawl_date", max_val="20221231")])
    for entries in result.values():
        for entry in entries:
            assert entry["crawl_date"] <= "20221231"


def test_all_predicates_combined(index):
    all_names = [r["digest"] for r in _RECORDS]
    result = index.search(
        all_names,
        [
            EqualityPredicate("sub_domain", "epa.gov"),
            RangePredicate("crawl_date", min_val="20220101", max_val="20231231"),
        ],
    )
    for entries in result.values():
        for entry in entries:
            assert entry["sub_domain"] == "epa.gov"
            assert "20220101" <= entry["crawl_date"] <= "20231231"


def test_predicate_no_matches_returns_empty(index):
    result = index.search(
        ["air_quality.pdf"], [EqualityPredicate("sub_domain", "nasa.gov")]
    )
    assert result == {}
