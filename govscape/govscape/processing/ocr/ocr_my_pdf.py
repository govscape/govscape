# AI modified: 2026-04-16, commit: 4a7a81118a560918e0ea7825dce27f3e5a60a340
"""
OCRMyPDF implementation for govscape processing.

This module provides an OCR implementation using the OCRMyPDF library.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union
import PyPDF2

from .abstract_ocr import AbstractOCR


class OCRMyPDF(AbstractOCR):
    """OCR implementation using OCRMyPDF library."""
    
    def __init__(self, **kwargs):
        """Initialize OCRMyPDF with optional configuration.
        
        Args:
            **kwargs: Configuration parameters for OCRMyPDF
        """
        super().__init__(**kwargs)
        try:
            import ocrmypdf
            self.ocrmypdf = ocrmypdf
        except ImportError:
            raise ImportError("OCRMyPDF not installed. Please install with: pip install ocrmypdf")
    
    def process_image(
        self, 
        image_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> str:
        """Process a single image and extract text using OCRMyPDF.
        
        Args:
            image_path: Path to the image file to process
            language: Optional language code for OCR
            
        Returns:
            Extracted text from the image
        """
        path = self._validate_file(image_path)
        
        # OCRMyPDF works with PDFs, so convert image to PDF first
        import img2pdf
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert image to PDF
            pdf_path = Path(temp_dir) / "temp.pdf"
            with open(pdf_path, 'wb') as f:
                f.write(img2pdf.convert(str(path)))
            
            # Process PDF with OCR
            output_pdf_path = Path(temp_dir) / "output.pdf"
            
            # Set OCR options
            ocr_options = {
                'language': language or self.config.get('language', 'eng'),
                'output_type': 'pdfa',
                'skip_text': True,
                'force_ocr': self.config.get('force_ocr', False),
                'optimize': self.config.get('optimize', 1),
                'jobs': self.config.get('jobs', 1),
            }
            
            # Perform OCR
            self.ocrmypdf.ocr(str(pdf_path), str(output_pdf_path), **ocr_options)
            
            # Extract text from the output PDF
            try:
                with open(output_pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() + '\n'
                return text.strip()
            except ImportError:
                raise ImportError("PyPDF2 not installed. Please install with: pip install PyPDF2")
    
    def process_pdf(
        self, 
        pdf_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> str:
        """Process a PDF document and extract text using OCRMyPDF.
        
        Args:
            pdf_path: Path to the PDF file to process
            language: Optional language code for OCR
            
        Returns:
            Extracted text from the PDF
        """
        path = self._validate_file(pdf_path)
        
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_pdf_path = Path(temp_dir) / "output.pdf"
            
            # Set OCR options
            ocr_options = {
                'language': language or self.config.get('language', 'eng'),
                'output_type': 'pdfa',
                'skip_text': True,
                'force_ocr': self.config.get('force_ocr', False),
                'optimize': self.config.get('optimize', 1),
                'jobs': self.config.get('jobs', 1),
            }
            
            # Perform OCR
            self.ocrmypdf.ocr(str(path), str(output_pdf_path), **ocr_options)
            
            # Extract text from the output PDF
            try:
                with open(output_pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() + '\n'
                return text.strip()
            except ImportError:
                raise ImportError("PyPDF2 not installed. Please install with: pip install PyPDF2")
    
    def process_pdf_per_page(
        self, 
        pdf_path: Union[str, Path], 
        language: Optional[str] = None
    ) -> List[str]:
        """Process a PDF document and extract text per page using OCRMyPDF.
        
        Args:
            pdf_path: Path to the PDF file to process
            language: Optional language code for OCR
            
        Returns:
            List of extracted text for each page
        """
        path = self._validate_file(pdf_path)
        
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_pdf_path = Path(temp_dir) / "output.pdf"
            
            # Set OCR options
            ocr_options = {
                'language': language or self.config.get('language', 'eng'),
                'output_type': 'pdfa',
                'skip_text': True,
                'force_ocr': self.config.get('force_ocr', False),
                'optimize': self.config.get('optimize', 1),
                'jobs': self.config.get('jobs', 1),
            }
            
            # Perform OCR
            self.ocrmypdf.ocr(str(path), str(output_pdf_path), **ocr_options)
            
            # Extract text per page from the output PDF
            try:
                with open(output_pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    page_texts = []
                    for page in reader.pages:
                        text = page.extract_text() or ''
                        page_texts.append(text.strip())
                return page_texts
            except ImportError:
                raise ImportError("PyPDF2 not installed. Please install with: pip install PyPDF2")
    
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
        """Get list of supported language codes for OCRMyPDF.
        
        Note: OCRMyPDF uses Tesseract language data files.
        
        Returns:
            List of supported language codes
        """
        # OCRMyPDF uses Tesseract, which supports many languages
        # Common language codes are listed here
        return [
            'eng', 'spa', 'fra', 'deu', 'ita', 'por', 'rus', 'jpn', 'kor',
            'chi_sim', 'chi_tra', 'ara', 'hin', 'tha', 'vie', 'tur', 'pol',
            'nld', 'swe', 'dan', 'nor', 'fin', 'ces', 'slk', 'hun', 'ron',
            'bul', 'hrv', 'ell', 'lit', 'lav', 'est', 'ukr', 'srp', 'slv',
            'cat', 'eus', 'glg', 'isl', 'gle', 'cym', 'bre', 'oci', 'cos',
            'ast', 'anp', 'awa', 'bho', 'mag', 'mai', 'new', 'sat', 'doi',
            'kok', 'gon', 'kfy', 'lah', 'pan', 'snd', 'skr', 'urdu'
        ]