# AI modified: 2026-04-16, commit: 4a7a81118a560918e0ea7825dce27f3e5a60a340
import json
import os
from multiprocessing import get_context
from typing import Optional, Type

import pypdfium2

from ..config import DataModel
from .ocr import AbstractOCR
from .processing_stage import ProcessingStage


def _convert_single_pdf(
    data_model, 
    pdf_file, 
    use_ocr: bool = False,
    ocr_class: Optional[Type[AbstractOCR]] = None,
    ocr_config: Optional[dict] = None,
    language: Optional[str] = None,
):
    """Process a single PDF file and extract text and images.
    
    Args:
        data_model: DataModel instance for file path generation
        pdf_file: Path to the PDF file to process
        use_ocr: Whether to use OCR for text extraction
        ocr_class: OCR class to use if use_ocr is True
        ocr_config: Configuration parameters for the OCR engine
        language: Language code for OCR
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
    os.makedirs(data_model.txt_pdf_directory(pdf_name), exist_ok=True)
    os.makedirs(data_model.img_pdf_directory(pdf_name), exist_ok=True)
    try:
        pdf = pypdfium2.PdfDocument(pdf_file)
        num_pages = len(pdf)
        gov_name = pdf.get_metadata_value("Title")

        json_data = {}
        timestamp = pdf.get_metadata_value("CreationDate")
        if len(gov_name) == 0:
            gov_name = "Unknown"
        if len(timestamp) == 0:
            timestamp = "Unknown"
        json_data["gov_name"] = gov_name
        json_data["timestamp"] = timestamp
        json_data["num_pages"] = num_pages
        os.makedirs(data_model.metadata_pdf_directory(pdf_name), exist_ok=True)
        with open(data_model.metadata_file_path(pdf_name), "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        images = []
        for i in range(num_pages):
            page = pdf[i]
            pil_image = page.render(scale=1.0).to_pil()
            images.append(pil_image)
        
        # Extract text either via OCR or directly from PDF
        if use_ocr and ocr_class is not None:
            # Use OCR for text extraction
            ocr_engine = ocr_class(**(ocr_config or {}))
            page_texts = ocr_engine.process_pdf_per_page(pdf_file, language)
        else:
            # Use direct text extraction from PDF
            page_texts = []
            for i in range(num_pages):
                page = pdf[i]
                page_text = page.get_textpage().get_text_bounded()
                page_texts.append(page_text)
    except Exception:
        return False

    for page_num, page_text in enumerate(page_texts):
        if page_text and len(page_text.strip()) != 0:
            with open(
                data_model.txt_page_path(pdf_name, page_num), "w", encoding="utf-8"
            ) as text_file:
                text_file.write(page_text)

        images[page_num].save(
            data_model.img_page_path(pdf_name, page_num), format="JPEG"
        )
    return True


class PDFExtractionStage(ProcessingStage):
    """Processing stage that extracts text and images from PDFs.
    
    This stage can optionally use OCR for text extraction instead of the default
    direct PDF text extraction. When OCR is enabled, each page is processed using
    the specified OCR engine and the text is saved per page following the 
    DATA_MODEL.md naming convention:
    - {test,dev,prod}-serving/txt/{digest}/{digest}_{pg_no}.txt
    - {test,dev,prod}-serving/img/{digest}/{digest}_{pg_no}.jpeg
    - {test,dev,prod}-serving/metadata/{digest}/metadata.json
    """
    
    def __init__(
        self, 
        data_model: DataModel, 
        pdf_files: list[str], 
        cpu_count: int,
        use_ocr: bool = False,
        ocr_class: Optional[Type[AbstractOCR]] = None,
        ocr_config: Optional[dict] = None,
        language: Optional[str] = None,
    ):
        """Initialize the PDF extraction stage.
        
        Args:
            data_model: DataModel instance for file path generation
            pdf_files: List of PDF file paths to process
            cpu_count: Number of CPU cores to use for parallel processing
            use_ocr: Whether to use OCR for text extraction (default: False)
            ocr_class: OCR class to use if use_ocr is True (default: None)
            ocr_config: Configuration parameters for the OCR engine (default: None)
            language: Language code for OCR (e.g., 'en', 'spa') (default: None)
        """
        self.data_model = data_model
        self.pdf_files = pdf_files
        self.cpu_count = cpu_count
        self.use_ocr = use_ocr
        self.ocr_class = ocr_class
        self.ocr_config = ocr_config or {}
        self.language = language

    def validate(self) -> None:
        """Validate that all PDF files exist."""
        missing = [f for f in self.pdf_files if not os.path.isfile(f)]
        if missing:
            raise ValueError(
                f"{len(missing)} PDF file(s) not found, e.g.: {missing[0]}"
            )

    def run(self) -> int:
        """Run the PDF extraction stage.
        
        Returns:
            Number of successfully processed PDF files
        """
        ctx = get_context("spawn")
        with ctx.Pool(processes=self.cpu_count) as pool:
            results = pool.starmap(
                _convert_single_pdf,
                [
                    (
                        self.data_model, 
                        pdf_file,
                        self.use_ocr,
                        self.ocr_class,
                        self.ocr_config,
                        self.language
                    )
                    for pdf_file in self.pdf_files
                ],
            )
        return sum(results)
