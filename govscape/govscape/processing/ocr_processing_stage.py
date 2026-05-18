"""OCR Processing Stage - Extracts text from PDF pages using OCR engines.

AI modified: 2026-05-15 unknown
"""

import logging
import os

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

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
            f"Must be one of: {list(ocr_engines.keys())}",
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
        if not CV2_AVAILABLE:
            raise ImportError(
                "cv2 (OpenCV) is required for OCR processing. "
                "Install it with: pip install opencv-python",
            )

        if not os.path.isdir(self.data_model.image_directory):
            raise ValueError(
                f"Image input directory does not exist: "
                f"{self.data_model.image_directory}",
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
                [f for f in os.listdir(digest_dir.path) if f.endswith(".jpeg")],
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

                    # Accumulate images for batch processing
                    if "_page_images" not in locals():
                        _page_images = []
                        _page_nums = []

                    _page_images.append(image)
                    _page_nums.append(
                        int(page_file.split("_")[-1].replace(".jpeg", ""))
                    )

                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    error_count += 1

            # If we collected page images, attempt batch OCR for the PDF
            if "_page_images" in locals() and _page_images:
                try:
                    # Prepare images according to engine expectations
                    try:
                        engine_name = self.ocr_engine.__class__.__name__.lower()
                    except Exception:
                        engine_name = ""

                    images_for_ocr = []
                    for img in _page_images:
                        if "paddle" in engine_name:
                            images_for_ocr.append(img)
                        else:
                            try:
                                images_for_ocr.append(
                                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                )
                            except Exception:
                                images_for_ocr.append(img)

                    # Try batch call: many OCR engines may accept a list of images.
                    try:
                        results = self.ocr_engine.extract_text(images_for_ocr)
                    except TypeError:
                        # Fallback to per-page calls if batch interface not supported
                        results = [
                            self.ocr_engine.extract_text(img) for img in images_for_ocr
                        ]

                    # Normalize results to a list of strings per page
                    if isinstance(results, str):
                        # Single string returned; assume single-page PDF
                        # or the same text applies to all pages
                        results_list = [results]
                    elif isinstance(results, list):
                        results_list = results
                    else:
                        # Best-effort string conversion per item
                        results_list = [str(r) for r in results]

                    # Ensure output directory exists for this PDF
                    txt_output_dir = self.data_model.txt_pdf_directory(digest)
                    os.makedirs(txt_output_dir, exist_ok=True)

                    # Write per-page text files
                    for idx, page_num in enumerate(_page_nums):
                        txt_output_path = self.data_model.txt_page_path(
                            digest, page_num
                        )
                        os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)
                        text_to_write = (
                            results_list[idx] if idx < len(results_list) else ""
                        )
                        with open(txt_output_path, "w", encoding="utf-8") as f:
                            f.write(text_to_write)
                        processed_count += 1
                        self.logger.debug(f"Processed: {txt_output_path}")

                    # Clean up locals for next digest
                    del _page_images
                    del _page_nums

                except Exception as e:
                    self.logger.error(f"Batch OCR processing failed for {digest}: {e}")
                    # If batch processing failed, count the pages we attempted
                    try:
                        error_count += len(_page_nums)
                    except Exception:
                        error_count += 1

        self.logger.info(
            f"OCR processing complete. Processed: {processed_count}, "
            f"Errors: {error_count}",
        )
