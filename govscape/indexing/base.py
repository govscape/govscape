from abc import ABC, abstractmethod
from typing import Any


class AbstractIndex(ABC):
    """Common contract for indexes that can provide ranked search results."""

    @abstractmethod
    def search(self, query: Any, k: int):
        """Return ranked results for a query."""

    @abstractmethod
    def total_entries(self) -> int:
        """Return the number of searchable entries."""
