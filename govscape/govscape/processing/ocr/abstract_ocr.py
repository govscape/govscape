# AI modified: 2026-04-16, commit: 4a7a81118a560918e0ea7825dce27f3e5a60a340
"""
Abstract OCR class for govscape processing.

This module defines the abstract base class that all OCR implementations
must inherit from.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union


class AbstractOCR(ABC):
    """Abstract base class for OCR implementations.
    
    This class defines the interface that all OCR engines must implement
    to be used within the govscape processing pipeline.
    """
    
    def __init__(self, **kwargs):
        """Initialize the OCR engine with optional configuration parameters."""
        self.config = kwargs
    
    @abstractmethod
    def process_image(
        self, 
        image_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> str:
        """Process a single image and extract text.
        
        Args:
            image_path: Path to the image file to process
            language: Optional language code for OCR (e.g., 'en', 'spa')
            
        Returns:
            Extracted text from the image
        """
        pass
    
    @abstractmethod
    def process_pdf(
        self, 
        pdf_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> str:
        """Process a PDF document and extract text.
        
        Args:
            pdf_path: Path to the PDF file to process
            language: Optional language code for OCR (e.g., 'en', 'spa')
            
        Returns:
            Extracted text from the PDF
        """
        pass
    
    @abstractmethod
    def process_batch(
        self, 
        file_paths: List[Union[str, Path]], 
        language: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Process multiple files and extract text from each.
        
        Args:
            file_paths: List of paths to image or PDF files
            language: Optional language code for OCR (e.g., 'en', 'spa')
            
        Returns:
            List of tuples containing (file_path, extracted_text)
        """
        pass
    
    @abstractmethod
    def process_pdf_per_page(
        self, 
        pdf_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> List[str]:
        """Process a PDF document and extract text per page.
        
        Args:
            pdf_path: Path to the PDF file to process
            language: Optional language code for OCR (e.g., 'en', 'spa')
            
        Returns:
            List of extracted text for each page
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes.
        
        Returns:
            List of supported language codes
        """
        pass
    
    def _validate_file(self, file_path: Union[str, Path]) -> Path:
        """Validate that a file exists and return a Path object.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Path object for the file
            
        Raises:
            FileNotFoundError: If the file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return path