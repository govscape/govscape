"""Tests for OCR implementations and OCRProcessingStage.

AI modified: 2026-05-15 unknown
"""

import os
import tempfile
from unittest.mock import patch

import pytest

import numpy as np

from govscape.config import DataModel
from govscape.processing import OCRProcessingStage
from govscape.processing.ocr import (
    EasyOCRImpl,
    OcrMyPDFImpl,
    OLMOcrImpl,
    PaddleOCRImpl,
)


def _create_test_image(text: str, size=(300, 80)) -> np.ndarray:
    """Create a simple white RGB image with black text.

    Returns a numpy array representing the image.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)
        return np.array(img)
    except Exception:
        # Fall back to a blank numpy image if PIL is not available
        return np.full((size[1], size[0], 3), 255, dtype=np.uint8)


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory for testing and yield DataModel."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_model = DataModel(tmpdir)
        os.makedirs(data_model.image_directory, exist_ok=True)
        yield tmpdir, data_model


OCR_IMPLS = [
    (EasyOCRImpl, {"languages": ["en"], "gpu": False}, "easyocr"),
    (PaddleOCRImpl, {"language": "en", "use_gpu": False}, "paddleocr"),
    (OLMOcrImpl, {"model_name": "default"}, "olmocr"),
    (OcrMyPDFImpl, {"language": "eng"}, None),
]


@pytest.mark.parametrize("impl_class,init_args,skip_pkg", OCR_IMPLS)
def test_ocr_implementations_on_sample_images(impl_class, init_args, skip_pkg):
    """Functionality-oriented test: each OCR implementation should extract expected text

    This test will be skipped for implementations whose dependencies are not installed.
    """
    # Skip based on known package name when provided
    if skip_pkg:
        pytest.importorskip(skip_pkg)

    # OcrMyPDF requires both ocrmypdf and pytesseract
    if impl_class is OcrMyPDFImpl:
        pytest.importorskip("ocrmypdf")
        pytest.importorskip("pytesseract")

    expected_pairs = [
        ("TEST ONE", _create_test_image("TEST ONE")),
        ("HELLO 123", _create_test_image("HELLO 123")),
    ]

    # Instantiate and validate the OCR engine
    ocr = impl_class(**init_args)
    try:
        ocr.validate()
    except ImportError:
        pytest.skip("Required OCR dependency is not available")

    # For each sample image, assert extracted text contains the expected substring
    for expected, img in expected_pairs:
        extracted = ocr.extract_text(img)
        assert isinstance(extracted, str)
        assert expected.lower().split()[0] in extracted.lower()


@pytest.mark.parametrize("impl_class,init_args,skip_pkg", OCR_IMPLS)
def test_ocr_processing_stage_writes_txt(
    impl_class, init_args, skip_pkg, temp_data_dir
):
    """Test OCRProcessingStage pipeline writes text files using different OCR engines.

    The OCR engine's actual `extract_text` is mocked to return deterministic text so the
    stage's file-writing behavior can be asserted for each implementation.
    """
    pytest.importorskip("cv2")

    tmpdir, data_model = temp_data_dir

    # Map impl_class to ocr_type string expected by OCRProcessingStage
    impl_to_type = {
        EasyOCRImpl: "easyocr",
        PaddleOCRImpl: "paddleocr",
        OLMOcrImpl: "olmocr",
        OcrMyPDFImpl: "ocrmypdf",
    }

    ocr_type = impl_to_type[impl_class]

    stage = OCRProcessingStage(data_model=data_model, ocr_type=ocr_type, **init_args)

    # Create a sample image on disk that cv2 can read
    import cv2

    digest = "abc123def456abc123def456abc123def45"
    img_dir = os.path.join(data_model.image_directory, digest)
    os.makedirs(img_dir, exist_ok=True)

    img = _create_test_image("PIPELINE TEST")
    img_path = os.path.join(img_dir, f"{digest}_0.jpeg")
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    mocked_text = "pipeline extracted text"

    # Patch engine methods to avoid external OCR dependencies during pipeline test
    with (
        patch.object(stage.ocr_engine, "validate", return_value=None),
        patch.object(stage.ocr_engine, "extract_text", return_value=mocked_text),
    ):
        # Validate and run stage
        stage.validate()
        stage.run()

    # Verify txt directory and file created with expected content
    txt_file = data_model.txt_page_path(digest, 0)
    assert os.path.exists(txt_file)
    with open(txt_file, encoding="utf-8") as f:
        content = f.read()
        assert content == mocked_text
