from abc import ABC, abstractmethod


class ProcessingStage(ABC):
    @abstractmethod
    def validate(self) -> None:
        """Raise ValueError if inputs are invalid."""

    @abstractmethod
    def run(self):
        pass
