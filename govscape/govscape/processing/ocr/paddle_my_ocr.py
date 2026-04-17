# AI modified: 2026-04-16, commit: 4a7a81118a560918e0ea7825dce27f3e5a60a340
"""
PaddleMyOCR implementation for govscape processing.

This module provides an OCR implementation using PaddleOCR.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

from .abstract_ocr import AbstractOCR


class PaddleMyOCR(AbstractOCR):
    """OCR implementation using PaddleOCR library."""
    
    def __init__(self, **kwargs):
        """Initialize PaddleMyOCR with optional configuration.
        
        Args:
            **kwargs: Configuration parameters for PaddleOCR
        """
        super().__init__(**kwargs)
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                lang=self.config.get('language', 'en'),
                use_gpu=self.config.get('gpu', False),
                use_angle_cls=self.config.get('use_angle_cls', True),
                det_model_dir=self.config.get('det_model_dir', None),
                rec_model_dir=self.config.get('rec_model_dir', None),
                cls_model_dir=self.config.get('cls_model_dir', None),
            )
        except ImportError:
            raise ImportError("PaddleOCR not installed. Please install with: pip install paddleocr")
    
    def process_image(
        self, 
        image_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> str:
        """Process a single image and extract text using PaddleOCR.
        
        Args:
            image_path: Path to the image file to process
            language: Optional language code for OCR
            
        Returns:
            Extracted text from the image
        """
        path = self._validate_file(image_path)
        
        # If language is specified, create a new OCR instance
        if language:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(lang=language, use_gpu=self.config.get('gpu', False))
        else:
            ocr = self.ocr
        
        # Perform OCR
        result = ocr.ocr(str(path), cls=True)
        
        # Extract text from results
        extracted_text = ''
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    extracted_text += line[1][0] + '\n'
        
        return extracted_text.strip()
    
    def process_pdf_per_page(
        self, 
        pdf_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> List[str]:
        """Process a PDF document and extract text per page using PaddleOCR.
        
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
    
    def process_pdf(
        self, 
        pdf_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> str:
        """Process a PDF document and extract text using PaddleOCR.
        
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
        for i, image in enumerate(images):
            # Save image temporarily to process
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name, 'PNG')
                text = self.process_image(temp_file.name, language)
                all_text.append(f"--- Page {i+1} ---\n{text}")
        
        return '\n\n'.join(all_text)
    
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
        """Get list of supported language codes for PaddleOCR.
        
        Returns:
            List of supported language codes
        """
        return [
            'en', 'ch', 'french', 'german', 'japan', 'korean', 'spanish', 
            'portuguese', 'russian', 'arabic', 'hindi', 'thai', 'vietnamese',
            'italian', 'dutch', 'polish', 'turkish', 'indonesian', 'malay',
            'czech', 'danish', 'finnish', 'norwegian', 'swedish', 'hungarian',
            'romanian', 'slovenian', 'croatian', 'serbian', 'bulgarian',
            'ukrainian', 'belarusian', 'macedonian', 'albanian', 'lithuanian',
            'latvian', 'estonian', 'irish', 'welsh', 'basque', 'catalan',
            'galician', 'maltese', 'afrikaans', 'swahili', 'amharic', 'hausa',
            'yoruba', 'zulu', 'xhosa', 'sotho', 'tswana', 'pedi', 'venda',
            'tsonga', 'ndebele', 'siSwati', 'sesotho', 'setswana', 'sepedi',
            'tshivenda', 'xitsonga', 'isindebele', 'siSwati'
        ]