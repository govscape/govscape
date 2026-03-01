import pytest

import numpy as np

from govscape.indexing import FAISSIndex, LanceDBVectorIndex


@pytest.mark.parametrize(
    ("index_name", "vector_index_class"),
    [
        ("faiss", FAISSIndex),
        ("lancedb", LanceDBVectorIndex),
    ],
    ids=["faiss", "lancedb"],
)
def test_vector_indices_round_trip(index_name, vector_index_class, tmp_path):
    embeddings = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.2, 0.1, 0.4, 0.3],
            [0.9, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    pdf_names = [
        "energy_policy.pdf",
        "transportation_update.pdf",
        "water_quality.pdf",
        "grid_resilience.pdf",
    ]
    pdf_pages = [3, 8, 5, 12]
    query_vector = embeddings[0]
    k_neighbors = 2

    index_dir = tmp_path / f"{index_name}_index"
    index = vector_index_class(index_dir.as_posix())

    index.add_batch(embeddings, pdf_names, pdf_pages)
    index.build_index()
    index.save_index()

    assert index.total_entries() == len(pdf_names)

    reloaded_index = vector_index_class(index_dir.as_posix())
    reloaded_index.load_index()

    search_result = reloaded_index.search(query_vector, k_neighbors)

    distances = np.asarray(search_result[0]).reshape(-1)
    assert distances.size == k_neighbors
    assert np.isfinite(distances).all()

    second_component = search_result[1]
    if isinstance(second_component, list):
        indices = [pdf_names.index(name) for name in second_component[:k_neighbors]]
    else:
        indices = [
            int(idx) for idx in np.asarray(second_component).reshape(-1)[:k_neighbors]
        ]

    expected_order = np.argsort(np.linalg.norm(embeddings - query_vector, axis=1))[
        :k_neighbors
    ].tolist()
    assert indices == expected_order

    if len(search_result) > 2:
        pages_component = search_result[2]
        expected_pages = [pdf_pages[idx] for idx in indices[: len(pages_component)]]
        assert list(pages_component)[: len(expected_pages)] == expected_pages

    assert reloaded_index.total_entries() == len(pdf_names)
