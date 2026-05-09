"""
Example: Using the OCR Module

This script demonstrates how to use the OCR processing module to extract
text from PDF page images and save them following the DATA_MODEL.md protocol.
"""

import logging

from govscape.config import DataModel
from govscape.processing import OCRProcessingStage

# Configure logging to see processing details
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def example_ocr_with_easyocr():
    """Example: Extract text using EasyOCR engine."""
    logger.info("Starting OCR extraction with EasyOCR...")

    # Initialize data model with your data directory
    data_model = DataModel("/path/to/data")

    # Create OCR processing stage with EasyOCR
    ocr_stage = OCRProcessingStage(
        data_model=data_model,
        ocr_type="easyocr",
        languages=["en"],  # Support English
        gpu=False,  # Set to True if GPU is available
    )

    # Validate OCR engine (checks installation and initializes)
    logger.info("Validating OCR engine...")
    ocr_stage.validate()

    # Run OCR on all page images
    logger.info("Processing images with OCR...")
    ocr_stage.run()

    logger.info("OCR extraction complete!")
    logger.info(f"Text files saved to: {data_model.txt_directory}")


def example_ocr_with_paddleocr():
    """Example: Extract text using PaddleOCR engine."""
    logger.info("Starting OCR extraction with PaddleOCR...")

    data_model = DataModel("/path/to/data")

    # Create OCR processing stage with PaddleOCR
    ocr_stage = OCRProcessingStage(
        data_model=data_model,
        ocr_type="paddleocr",
        language="en",
        use_gpu=False,  # Set to True if GPU is available
    )

    logger.info("Validating OCR engine...")
    ocr_stage.validate()

    logger.info("Processing images with OCR...")
    ocr_stage.run()

    logger.info("OCR extraction complete!")


def example_ocr_with_easyocr_multilingual():
    """Example: Extract text from multilingual documents."""
    logger.info("Starting OCR extraction with multilingual support...")

    data_model = DataModel("/path/to/data")

    # Support multiple languages
    ocr_stage = OCRProcessingStage(
        data_model=data_model,
        ocr_type="easyocr",
        languages=["en", "fr", "es", "de"],  # English, French, Spanish, German
        gpu=True,  # Recommended for multiple languages
    )

    logger.info("Validating OCR engine...")
    ocr_stage.validate()

    logger.info("Processing images with OCR...")
    ocr_stage.run()

    logger.info("Multilingual OCR extraction complete!")


def example_ocr_with_tesseract():
    """Example: Extract text using OcrMyPDF/Tesseract engine."""
    logger.info("Starting OCR extraction with OcrMyPDF/Tesseract...")

    data_model = DataModel("/path/to/data")

    # Create OCR processing stage with OcrMyPDF/Tesseract
    ocr_stage = OCRProcessingStage(
        data_model=data_model,
        ocr_type="ocrmypdf",
        language="eng",  # Tesseract language code
    )

    logger.info("Validating OCR engine...")
    ocr_stage.validate()

    logger.info("Processing images with OCR...")
    ocr_stage.run()

    logger.info("Tesseract OCR extraction complete!")


def example_direct_ocr_usage():
    """Example: Use OCR engine directly without processing stage."""
    import cv2

    from govscape.processing.ocr import EasyOCRImpl

    logger.info("Direct OCR engine usage example...")

    # Create and validate OCR engine
    ocr = EasyOCRImpl(languages=["en"], gpu=False)
    ocr.validate()

    # Load and process a single image
    image_path = "/path/to/page_image.jpeg"
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract text
    text = ocr.extract_text(image_rgb)

    logger.info(f"Extracted text:\n{text}")

    # Save text to file (manual example)
    output_path = "/path/to/output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    logger.info(f"Text saved to: {output_path}")


def example_batch_processing():
    """Example: Process a batch of PDFs with progress tracking."""
    import os

    logger.info("Batch processing example...")

    data_model = DataModel("/path/to/data")

    # Create OCR processing stage
    ocr_stage = OCRProcessingStage(
        data_model=data_model,
        ocr_type="easyocr",
        languages=["en"],
        gpu=False,
    )

    ocr_stage.validate()

    # Count total images to process
    total_images = 0
    for digest_dir in os.scandir(data_model.image_directory):
        if digest_dir.is_dir():
            total_images += len(
                [f for f in os.listdir(digest_dir.path) if f.endswith(".jpeg")]
            )

    logger.info(f"Total images to process: {total_images}")

    # Run processing
    ocr_stage.run()

    logger.info("Batch processing complete!")


if __name__ == "__main__":
    # Choose which example to run by uncommenting one of the following:

    # Basic EasyOCR example
    # example_ocr_with_easyocr()

    # PaddleOCR example
    # example_ocr_with_paddleocr()

    # Multilingual support
    # example_ocr_with_easyocr_multilingual()

    # Tesseract/OcrMyPDF example
    # example_ocr_with_tesseract()

    # Direct OCR engine usage
    # example_direct_ocr_usage()

    # Batch processing
    # example_batch_processing()

    logger.info("Please uncomment one of the example functions to run!")
