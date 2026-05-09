"""OCR Processing Stage - Extracts text from PDF pages using OCR engines."""

import logging
import os

import cv2

from ..config import DataModel
from .ocr.base_ocr import BaseOCR
from .processing_stage import ProcessingStage


def _build_ocr_engine(ocr_type: str, **kwargs) -> BaseOCR:
    """Factory function to build OCR engines.

    Args:
        ocr_type: Type of OCR engine ('easyocr', 'paddleocr', 'olmocr', 'ocrmypdf').
        **kwargs: Additional arguments to pass to the OCR engine constructor.

    Returns:
        An initialized OCR engine.

    Raises:
        ValueError: If ocr_type is not supported.
    """
    from .ocr import EasyOCRImpl, OcrMyPDFImpl, OLMOcrImpl, PaddleOCRImpl

    ocr_engines = {
        "easyocr": EasyOCRImpl,
        "paddleocr": PaddleOCRImpl,
        "olmocr": OLMOcrImpl,
        "ocrmypdf": OcrMyPDFImpl,
    }

    if ocr_type not in ocr_engines:
        raise ValueError(
            f"Unsupported OCR type: {ocr_type}. "
            f"Must be one of: {list(ocr_engines.keys())}"
        )

    engine_class = ocr_engines[ocr_type]
    return engine_class(**kwargs)


class OCRProcessingStage(ProcessingStage):
    """Processing stage that performs OCR on PDF page images.

    This stage:
    1. Reads page images from {image_directory}/{digest}/{digest}_{pg_no}.jpeg
    2. Applies OCR using the specified engine
    3. Saves extracted text to {txt_directory}/{digest}/{digest}_{pg_no}.txt

    Following the DATA_MODEL.md protocol for text file organization.
    """

    def __init__(self, data_model: DataModel, ocr_type: str = "easyocr", **ocr_kwargs):
        """Initialize the OCR Processing Stage.

        Args:
            data_model: DataModel instance defining directory structure.
            ocr_type: Type of OCR engine to use (default: 'easyocr').
            **ocr_kwargs: Additional arguments to pass to the OCR engine.
                For EasyOCR: languages=['en', ...], gpu=False
                For PaddleOCR: language='en', use_gpu=False
                For OLMOcr: model_name='default'
                For OcrMyPDF: language='eng', output_type='txt'
        """
        self.data_model = data_model
        self.ocr_engine = _build_ocr_engine(ocr_type, **ocr_kwargs)
        self.logger = logging.getLogger(__name__)

    def validate(self) -> None:
        """Validate that the image directory exists and OCR engine is initialized."""
        if not os.path.isdir(self.data_model.image_directory):
            raise ValueError(
                f"Image input directory does not exist: "
                f"{self.data_model.image_directory}"
            )

        try:
            self.ocr_engine.validate()
        except Exception as e:
            raise ValueError(f"OCR engine validation failed: {e}") from e

    def run(self):
        """Run OCR on all PDF page images and save extracted text.

        Processes each PDF's pages and saves text in the format:
        {txt_directory}/{digest}/{digest}_{pg_no}.txt
        """
        os.makedirs(self.data_model.txt_directory, exist_ok=True)

        processed_count = 0
        error_count = 0

        # Iterate through digest directories in the image directory
        for digest_dir in os.scandir(self.data_model.image_directory):
            if not digest_dir.is_dir():
                continue

            digest = digest_dir.name
            txt_output_dir = self.data_model.txt_pdf_directory(digest)
            os.makedirs(txt_output_dir, exist_ok=True)

            # Process each page image
            page_files = sorted(
                [f for f in os.listdir(digest_dir.path) if f.endswith(".jpeg")]
            )

            for page_file in page_files:
                try:
                    image_path = os.path.join(digest_dir.path, page_file)

                    # Read image using cv2
                    image = cv2.imread(image_path)
                    if image is None:
                        self.logger.warning(f"Failed to read image: {image_path}")
                        error_count += 1
                        continue

                    # Convert BGR to RGB for some OCR engines
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Extract page number from filename (e.g., "digest_0.jpeg" -> 0)
                    page_num = int(page_file.split("_")[-1].replace(".jpeg", ""))

                    # Perform OCR
                    extracted_text = self.ocr_engine.extract_text(image_rgb)

                    # Save text to file following DATA_MODEL.md protocol
                    txt_output_path = self.data_model.txt_page_path(digest, page_num)
                    os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

                    with open(txt_output_path, "w", encoding="utf-8") as f:
                        f.write(extracted_text)

                    processed_count += 1
                    self.logger.debug(f"Processed: {txt_output_path}")

                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    error_count += 1

        self.logger.info(
            f"OCR processing complete. Processed: {processed_count}, "
            f"Errors: {error_count}"
        )
