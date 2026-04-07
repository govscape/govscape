from abc import ABC, abstractmethod
import time

class OCREngine(ABC):
    """Abstract base class for all OCR engines."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, input_path: str) -> str:
        """
        Executes the OCR process.
        Returns: Extracted text/markdown string.
        """
        pass

    def run_benchmark(self, input_path: str) -> dict:
        """
        Wrapper to handle timing and error reporting consistently.
        """
        try:
            start = time.perf_counter()
            content = self.process(input_path)
            duration = time.perf_counter() - start
            return {
                'time': duration,
                'content': content,
                'status': 'Success'
            }
        except Exception as e:
            return {
                'time': 0,
                'content': f"Error: {str(e)}",
                'status': 'Failed'
            }