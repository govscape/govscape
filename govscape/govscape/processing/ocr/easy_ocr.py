# AI modified: 2026-04-16, commit: 4a7a81118a560918e0ea7825dce27f3e5a60a340
"""
EasyOCR implementation for govscape processing.

This module provides an OCR implementation using the EasyOCR library.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from .abstract_ocr import AbstractOCR


class EasyOCR(AbstractOCR):
    """OCR implementation using EasyOCR library."""
    
    def __init__(self, **kwargs):
        """Initialize EasyOCR with optional configuration.
        
        Args:
            **kwargs: Configuration parameters for EasyOCR
        """
        super().__init__(**kwargs)
        try:
            import easyocr
            self.reader = easyocr.Reader(
                lang_list=self.config.get('languages', ['en']),
                gpu=self.config.get('gpu', False),
                model_storage_directory=self.config.get('model_dir', None),
                download_enabled=self.config.get('download_enabled', True)
            )
        except ImportError:
            raise ImportError("EasyOCR not installed. Please install with: pip install easyocr")
    
    def process_image(
        self, 
        image_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> str:
        """Process a single image and extract text using EasyOCR.
        
        Args:
            image_path: Path to the image file to process
            language: Optional language code for OCR
            
        Returns:
            Extracted text from the image
        """
        path = self._validate_file(image_path)
        
        # If language is specified, create a new reader with that language
        if language:
            import easyocr
            reader = easyocr.Reader([language], gpu=self.config.get('gpu', False))
        else:
            reader = self.reader
        
        # Perform OCR
        results = reader.readtext(str(path))
        
        # Extract text from results
        extracted_text = ' '.join([result[1] for result in results])
        
        return extracted_text
    
    def process_pdf(
        self, 
        pdf_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> str:
        """Process a PDF document and extract text using EasyOCR.
        
        Args:
            pdf_path: Path to the PDF file to process
            language: Optional language code for OCR
            
        Returns:
            Extracted text from the PDF
        """
        path = self._validate_file(pdf_path)
        
        # Convert PDF to images and process each page
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("pdf2image not installed. Please install with: pip install pdf2image")
        
        # Convert PDF to images
        images = convert_from_path(str(path))
        
        # Process each page
        all_text = []
        for image in images:
            # Save image temporarily to process
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name, 'PNG')
                text = self.process_image(temp_file.name, language)
                all_text.append(text)
        
        return '\n\n'.join(all_text)
    
    def process_pdf_per_page(
        self, 
        pdf_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> List[str]:
        """Process a PDF document and extract text per page using EasyOCR.
        
        Args:
            pdf_path: Path to the PDF file to process
            language: Optional language code for OCR
            
        Returns:
            List of extracted text for each page
        """
        path = self._validate_file(pdf_path)
        
        # Convert PDF to images and process each page
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("pdf2image not installed. Please install with: pip install pdf2image")
        
        # Convert PDF to images
        images = convert_from_path(str(path))
        
        # Process each page
        page_texts = []
        for image in images:
            # Save image temporarily to process
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name, 'PNG')
                text = self.process_image(temp_file.name, language)
                page_texts.append(text.strip())
        
        return page_texts
    
    def process_batch(
        self, 
        file_paths: List[Union[str, Path]], 
        language: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """Process multiple files and extract text from each.
        
        Args:
            file_paths: List of paths to image or PDF files
            language: Optional language code for OCR
            
        Returns:
            List of tuples containing (file_path, extracted_text)
        """
        results = []
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.suffix.lower() == '.pdf':
                    text = self.process_pdf(file_path, language)
                else:
                    text = self.process_image(file_path, language)
                results.append((str(file_path), text))
            except Exception as e:
                results.append((str(file_path), f"Error: {str(e)}"))
        return results
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes for EasyOCR.
        
        Returns:
            List of supported language codes
        """
        # EasyOCR supports many languages, returning common ones
        return [
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'fr', 'de', 'es', 'it', 'pt',
            'ru', 'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no',
            'fi', 'cs', 'sk', 'hu', 'ro', 'bg', 'hr', 'el', 'lt', 'lv', 'et'
        ]