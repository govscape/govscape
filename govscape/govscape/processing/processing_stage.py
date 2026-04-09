from abc import ABC, abstractmethod


class ProcessingStage(ABC):
    @abstractmethod
    def validate(self) -> list[str]:
        """Return a list of validation error messages. Empty list means valid."""

    @abstractmethod
    def run(self):
        pass
