# AI modified
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .keyword import AbstractKeywordIndex
from .metadata import AbstractMetadataIndex
from .vector import AbstractVectorIndex

STRATEGY_PREFILTER = "prefilter"
STRATEGY_POSTFILTER = "postfilter"


@dataclass
class HybridSearchState:
    strategy: str
    current_k: int
    estimated_selectivity: float
    prefilter_cost: float
    postfilter_cost: float


class AbstractHybridMetadataIndex(ABC):
    def __init__(self, metadata_index: AbstractMetadataIndex):
        self.metadata_index = metadata_index

    @staticmethod
    def deduplicate_by_digest(
        distances: list[float], digests: list[str], pages: list[str]
    ) -> list[tuple[float, str, str]]:
        seen: set[str] = set()

        deduped: list[tuple[float, str, str]] = []

        for distance, digest, page in zip(distances, digests, pages, strict=False):
            if digest in seen:
                continue

            seen.add(digest)
            deduped.append((float(distance), digest, str(page)))

        return deduped

    @staticmethod
    def _apply_blacklist(rows: list[tuple[float, str, str]], blacklist: set[str]):
        return [row for row in rows if row[1] not in blacklist]

    def _metadata_size(self) -> int:
        return self.metadata_index.total_entries()

    def _estimate_selectivity(self, predicates) -> float:
        if not predicates:
            return 1.0
        return self.metadata_index.estimate_selectivity(predicates)

    def _choose_strategy(
        self, estimated_selectivity: float, target_results: int
    ) -> tuple[str, float, float]:

        # Ensure selectivity is not too close to zero.
        safe_selectivity: float = max(float(estimated_selectivity), 1e-6)

        # Cost model:
        # cost of prefiltering  = selectivity * size of metadata database
        # cost of postfiltering = k * 1/selectivity

        prefilter_cost: float = safe_selectivity * float(self._metadata_size())
        postfilter_cost: float = float(target_results) * (1.0 / safe_selectivity)

        strategy = (
            STRATEGY_PREFILTER
            if prefilter_cost <= postfilter_cost
            else STRATEGY_POSTFILTER
        )
        return strategy, prefilter_cost, postfilter_cost

    @abstractmethod
    def _index_total_entries(self) -> int:
        pass

    @abstractmethod
    def _run_prefilter(
        self,
        query_embedding,
        predicates,
        target_results,
        candidates,
    ):
        pass

    @abstractmethod
    def _run_postfilter(
        self, query_embedding, predicates, target_results, blacklist, selectivity
    ):
        pass

    def search(
        self,
        query_embedding,
        predicates,
        target_results,
        blacklist=None,
    ):
        if blacklist is None:
            blacklist = set()

        estimated_selectivity = self._estimate_selectivity(predicates)

        strategy, prefilter_cost, postfilter_cost = self._choose_strategy(
            estimated_selectivity, target_results
        )

        rows = []
        metadata = {}
        current_k = 0

        if strategy == STRATEGY_PREFILTER:
            candidates = self.metadata_index.get_candidate_digests(predicates)
            candidates = candidates.difference(blacklist)

            rows, metadata, current_k = self._run_prefilter(
                query_embedding,
                predicates,
                target_results,
                candidates,
            )

        if strategy == STRATEGY_POSTFILTER:
            rows, metadata, current_k = self._run_postfilter(
                query_embedding,
                predicates,
                target_results,
                blacklist,
                estimated_selectivity,
            )

        return (
            rows,
            metadata,
            HybridSearchState(
                strategy=strategy,
                current_k=current_k,
                estimated_selectivity=estimated_selectivity,
                prefilter_cost=prefilter_cost,
                postfilter_cost=postfilter_cost,
            ),
        )


class HybridVectorMetadataIndex(AbstractHybridMetadataIndex):
    def __init__(
        self,
        vector_index: AbstractVectorIndex,
        metadata_index: AbstractMetadataIndex,
        vector_store_key: str = "text",
    ):
        super().__init__(metadata_index=metadata_index)
        self.vector_index = vector_index
        self.vector_store_key = vector_store_key

    def _index_total_entries(self) -> int:
        return self.vector_index.total_entries()

    def _run_prefilter(self, query_embedding, predicates, target_results, candidates):
        vectors, digests, pages = self.metadata_index.get_vectors_for_digests(
            self.vector_store_key, candidates
        )
        if len(digests) == 0:
            return [], {}, 0

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        if query_vec.ndim == 2:
            query_vec = query_vec[0]
        if query_vec.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector or shape (1, d)")

        distances = np.linalg.norm(vectors - query_vec, axis=1)
        order = np.argsort(distances)
        ranked_distances = [float(distances[i]) for i in order]
        ranked_digests = [digests[i] for i in order]
        ranked_pages = [str(pages[i]) for i in order]
        ranked_rows = self.deduplicate_by_digest(
            ranked_distances,
            ranked_digests,
            ranked_pages,
        )

        selected_rows = ranked_rows[:target_results]
        selected_digests = [digest for _, digest, _ in selected_rows]
        metadata = self.metadata_index.search(selected_digests, predicates)
        return selected_rows, metadata, len(digests)

    def _run_postfilter(
        self,
        query_embedding,
        predicates,
        target_results,
        blacklist,
        selectivity,
    ):
        safe_selectivity = max(selectivity, 1e-6)
        current_k = int(math.ceil(target_results * (1.0 / safe_selectivity)))
        old_results_found = -1
        filtered_rows = []
        metadata = {}

        while len(filtered_rows) < target_results:
            distances, digests, pages = self.vector_index.search(
                query_embedding, current_k
            )
            deduped = self.deduplicate_by_digest(distances, digests, pages)
            deduped = self._apply_blacklist(deduped, blacklist)
            candidate_digests = [digest for _, digest, _ in deduped]
            metadata = self.metadata_index.search(candidate_digests, predicates)
            filtered_rows = [row for row in deduped if row[1] in metadata]

            if len(filtered_rows) >= target_results:
                break
            if current_k >= self._index_total_entries():
                break

            results_found = len(filtered_rows)
            if results_found == old_results_found and not predicates:
                break
            old_results_found = results_found

            current_k = min(self._index_total_entries(), current_k * 2)
            if (
                current_k == self._index_total_entries()
                and len(filtered_rows) == old_results_found
            ):
                break

        return filtered_rows, metadata, current_k


class HybridKeywordMetadataIndex(AbstractHybridMetadataIndex):
    def __init__(
        self, keyword_index: AbstractKeywordIndex, metadata_index: AbstractMetadataIndex
    ):
        super().__init__(metadata_index=metadata_index)
        self.keyword_index = keyword_index

    def _index_total_entries(self) -> int:
        return self.keyword_index.total_entries()

    def _run_prefilter(
        self,
        query_text,
        predicates,
        target_results,
        candidates,
    ):
        if not candidates:
            return [], {}, 0

        current_k = max(1, target_results)
        old_results_found = -1
        filtered_rows = []
        metadata = {}

        # Since we cannot know in advance how many results will remain after
        # deduplication, we still need a loop for keyword prefiltering.
        while len(filtered_rows) < target_results:
            distances, digests, pages = self.keyword_index.search_filtered(
                query_text,
                current_k,
                candidates,
            )
            # Keep document-level uniqueness in ranking output.
            deduped = self.deduplicate_by_digest(distances, digests, pages)
            candidate_digests = [digest for _, digest, _ in deduped]
            metadata = self.metadata_index.search(candidate_digests, predicates)
            filtered_rows = [row for row in deduped if row[1] in metadata]

            if len(filtered_rows) >= target_results:
                break
            if current_k >= self._index_total_entries():
                break

            results_found = len(filtered_rows)
            if results_found == old_results_found and not predicates:
                break
            old_results_found = results_found

            current_k = min(self._index_total_entries(), current_k * 2)

        return filtered_rows, metadata, current_k

    def _run_postfilter(
        self, query_text, predicates, target_results, blacklist, selectivity
    ):
        safe_selectivity = max(selectivity, 1e-6)
        current_k = int(math.ceil(target_results * (1.0 / safe_selectivity)))
        old_results_found = -1
        filtered_rows = []
        metadata = {}

        while len(filtered_rows) < target_results:
            distances, digests, pages = self.keyword_index.search(query_text, current_k)
            deduped = self.deduplicate_by_digest(distances, digests, pages)
            deduped = self._apply_blacklist(deduped, blacklist)
            candidate_digests = [digest for _, digest, _ in deduped]
            metadata = self.metadata_index.search(candidate_digests, predicates)
            filtered_rows = [row for row in deduped if row[1] in metadata]

            if len(filtered_rows) >= target_results:
                break

            results_found = len(filtered_rows)
            if results_found == old_results_found and not predicates:
                break
            old_results_found = results_found
            current_k = current_k * 2

            if current_k > self._index_total_entries():
                break

        return filtered_rows, metadata, current_k
