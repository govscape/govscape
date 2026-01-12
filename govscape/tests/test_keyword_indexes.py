import pytest

from govscape.indexing import LanceDBKeywordIndex, SQLiteKeywordIndex, WhooshKeywordIndex


@pytest.fixture
def sample_documents():
    texts = [
        "Offshore wind procurement policy overview with federal incentives,",
        "Coastal resilience strategy for climate change adaptation planning&",
        "Transportation infrastructure funding updates and grants!",
        "Wildfire mitigation tactics for rural counties*",
        "Solar energy adoption roadmap for municipal buildings;",
        "Water conservation guidelines for drought-prone regions^",
        "Wastewater treatment modernization feasibility study",
        "Broadband expansion program for underserved communities",
        "Emergency management coordination best practices",
        "Public health outreach during extreme heat events",
        "Affordable housing development incentives summary",
        "Urban forestry initiatives to reduce heat islands",
        "Agricultural runoff reduction and watershed protection",
        "Electric vehicle charging infrastructure deployment plan",
        "Maritime port emissions reduction and monitoring",
    ]
    pdf_names = [
        "wind_policy.pdf",
        "coastal_resilience.pdf",
        "transport_funding.pdf",
        "wildfire_mitigation.pdf",
        "solar_roadmap.pdf",
        "water_conservation.pdf",
        "wastewater_modernization.pdf",
        "broadband_expansion.pdf",
        "emergency_management.pdf",
        "public_health_heat.pdf",
        "affordable_housing.pdf",
        "urban_forestry.pdf",
        "agricultural_runoff.pdf",
        "ev_infrastructure.pdf",
        "maritime_emissions.pdf",
    ]
    pages = [5, 12, 3, 7, 9, 6, 4, 8, 10, 11, 13, 2, 14, 15, 1]
    return texts, pdf_names, pages


@pytest.mark.parametrize(
    "index_cls",
    [LanceDBKeywordIndex, SQLiteKeywordIndex, WhooshKeywordIndex],
    ids=["lancedb", "sqlite", "whoosh"],
)
def test_keyword_indexes_round_trip(tmp_path, index_cls, sample_documents):
    index_dir = tmp_path / "keyword_index"
    texts, pdf_names, pages = sample_documents

    index = index_cls(index_dir.as_posix())
    index.build_index()
    index.add_batch(texts, pdf_names, pages)
    index.save_index()

    assert index.total_entries() == len(texts)

    reloaded_index = index_cls(index_dir.as_posix())
    reloaded_index.load_index()

    assert reloaded_index.total_entries() == len(texts)

    expected_term_hits = [
        name
        for text, name in zip(texts, pdf_names)
        if "resilience" in text.lower()
    ]
    scores, found_pdfs, found_pages = reloaded_index.search("resilience", k=5)

    assert len(found_pdfs) == len(expected_term_hits)
    assert pdf_names[1] in found_pdfs
    assert all(page.isdigit() for page in found_pages)

    phrase_scores, phrase_pdfs, phrase_pages = reloaded_index.search(
        '"coastal resilience"', k=2
    )

    assert len(phrase_pdfs) == 1
    assert phrase_pdfs[0] == pdf_names[1]
    assert phrase_pages[0] == str(pages[1])
