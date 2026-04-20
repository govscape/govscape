# AI modified: 2026-04-19 21:12:31 c1b6021e
# AI modified: 2026-04-20 00:00:00 c1b6021e
# AI modified: 2026-04-20 00:00:00 c1b6021e
# AI modified: 2026-04-20 00:00:00 c1b6021e
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .metadata import AbstractMetadataIndex
from .vector import AbstractVectorIndex

STRATEGY_PREFILTER = "prefilter"
STRATEGY_POSTFILTER = "postfilter"


@dataclass
class HybridSearchState:
    strategy: str
    current_k: int


class HybridVectorMetadataIndex:
    def __init__(
        self,
        vector_index: AbstractVectorIndex,
        metadata_index: AbstractMetadataIndex,
        prefilter_threshold: float = 0.2,
        vector_store_key: str = "text",
    ):
        self.vector_index = vector_index
        self.metadata_index = metadata_index
        self.prefilter_threshold = prefilter_threshold
        self.vector_store_key = vector_store_key

    @staticmethod
    def deduplicate_by_pdf(distances, names, pages):
        seen = set()
        deduped = []
        for distance, name, page in zip(distances, names, pages, strict=False):
            if name in seen:
                continue
            seen.add(name)
            deduped.append((float(distance), name, str(page)))
        return deduped

    def _choose_strategy(self, predicates) -> str:
        if not predicates:
            return STRATEGY_POSTFILTER
        if not hasattr(self.metadata_index, "estimate_selectivity"):
            return STRATEGY_POSTFILTER
        estimated = self.metadata_index.estimate_selectivity(predicates)
        if estimated <= self.prefilter_threshold:
            return STRATEGY_PREFILTER
        return STRATEGY_POSTFILTER

    def _apply_blacklist(self, rows, blacklist):
        if not blacklist:
            return rows
        return [row for row in rows if row[1] not in blacklist]

    def _run_prefilter(self, query_embedding, predicates, target_results, blacklist):
        if not hasattr(self.metadata_index, "get_candidate_pdf_names"):
            return [], {}, 0
        if not hasattr(self.metadata_index, "get_vectors_for_pdf_names"):
            return [], {}, 0

        candidates = self.metadata_index.get_candidate_pdf_names(predicates)
        if blacklist:
            candidates = candidates.difference(blacklist)
        if not candidates:
            return [], {}, 0

        vectors, names, pages = self.metadata_index.get_vectors_for_pdf_names(
            self.vector_store_key,
            candidates,
        )
        if len(names) == 0:
            return [], {}, 0

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        if query_vec.ndim == 2:
            query_vec = query_vec[0]
        if query_vec.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector or shape (1, d)")

        distances = np.linalg.norm(vectors - query_vec, axis=1)
        order = np.argsort(distances)
        ranked_rows = [(float(distances[i]), names[i], str(pages[i])) for i in order]
        ranked_rows = self._apply_blacklist(ranked_rows, blacklist)

        selected_rows = ranked_rows[:target_results]
        selected_names = [name for _, name, _ in selected_rows]
        metadata = self.metadata_index.search(selected_names, predicates)
        return selected_rows, metadata, len(names)

    def _run_postfilter(
        self,
        query_embedding,
        predicates,
        target_results,
        blacklist,
        selectivity,
        max_k,
    ):
        safe_selectivity = max(selectivity, 1e-6)
        initial_k = int(math.ceil(target_results * (1.0 / safe_selectivity)))
        current_k = max(1, min(max_k, initial_k))
        old_results_found = -1
        filtered_rows = []
        metadata = {}

        while len(filtered_rows) < target_results:
            distances, names, pages = self.vector_index.search(
                query_embedding, current_k
            )
            deduped = self.deduplicate_by_pdf(distances, names, pages)
            deduped = self._apply_blacklist(deduped, blacklist)
            candidate_names = [name for _, name, _ in deduped]
            metadata = self.metadata_index.search(candidate_names, predicates)
            filtered_rows = [row for row in deduped if row[1] in metadata]

            if len(filtered_rows) >= target_results:
                break
            if current_k >= max_k:
                break

            results_found = len(filtered_rows)
            if results_found == old_results_found and not predicates:
                break
            old_results_found = results_found

            current_k = min(max_k, current_k * 2)
            if current_k == max_k and len(filtered_rows) == old_results_found:
                break

        return filtered_rows, metadata, current_k

    def search(
        self,
        query_embedding,
        predicates,
        target_results,
        blacklist=None,
        max_k=None,
    ):
        if blacklist is None:
            blacklist = set()

        index_limit = max(1, self.vector_index.total_entries())
        postfilter_max_k = min(max_k or 100000, index_limit)
        strategy = self._choose_strategy(predicates)
        estimated_selectivity = 1.0
        if predicates and hasattr(self.metadata_index, "estimate_selectivity"):
            estimated_selectivity = max(
                0.0,
                min(1.0, float(self.metadata_index.estimate_selectivity(predicates))),
            )

        if strategy == STRATEGY_PREFILTER:
            rows, metadata, current_k = self._run_prefilter(
                query_embedding,
                predicates,
                target_results,
                blacklist,
            )
        else:
            rows, metadata, current_k = self._run_postfilter(
                query_embedding,
                predicates,
                target_results,
                blacklist,
                estimated_selectivity,
                postfilter_max_k,
            )

        return rows, metadata, HybridSearchState(strategy=strategy, current_k=current_k)
