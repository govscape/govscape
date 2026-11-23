"""Tests for OCR implementations and OCRProcessingStage."""

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


class _StaticTextOCR:
    """Fake OCR engine returning a fixed string for every image.

    Lets us exercise the OCRProcessingStage orchestration (batching, worker
    dispatch, metadata mapping, writing) without loading a real OCR model.
    """

    TEXT = "fake-ocr-text"

    def validate(self) -> None:
        return None

    def extract_text(self, images):
        return [self.TEXT for _ in images]


class _InlineProcessPoolExecutor:
    """Drop-in ProcessPoolExecutor substitute that runs everything in-process.

    It honors ``initializer``/``initargs`` exactly like the real executor, so the
    per-worker engine setup in ``run`` is exercised, while avoiding the
    spawning of real processes (and loading of real OCR models) during tests.
    """

    map_calls = 0

    def __init__(
        self, max_workers=None, mp_context=None, initializer=None, initargs=()
    ):
        self._initializer = initializer
        self._initargs = initargs

    def __enter__(self):
        if self._initializer is not None:
            self._initializer(*self._initargs)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, items):
        type(self).map_calls += 1
        return [fn(item) for item in items]


OCR_IMPLS = [
    (EasyOCRImpl, {"languages": ["en"], "gpu": False}, "easyocr"),
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

    # olmOCR runs a ~7B vision-language model (large download, slow, GPU-bound),
    # so its real-inference test is opt-in rather than run on every invocation.
    if impl_class is OLMOcrImpl and not os.environ.get("RUN_OLMOCR_MODEL"):
        pytest.skip("Set RUN_OLMOCR_MODEL=1 to run the olmOCR model test.")

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

    # Extract the whole batch at once and assert each result contains its substring
    images = [img for _expected, img in expected_pairs]
    extracted = ocr.extract_text(images)
    assert isinstance(extracted, list)
    assert len(extracted) == len(expected_pairs)
    for (expected, _img), text in zip(expected_pairs, extracted, strict=True):
        assert isinstance(text, str)
        assert expected.lower().split()[0] in text.lower()


@pytest.mark.parametrize("impl_class,init_args,skip_pkg", OCR_IMPLS)
def test_ocr_processing_stage_writes_txt(
    impl_class, init_args, skip_pkg, temp_data_dir
):
    """OCRProcessingStage should dispatch batches through a ProcessPoolExecutor and
    write one txt file per page, for every OCR engine type.

    The worker-side engine is replaced with a fake so no real OCR model is loaded,
    and the pool runs inline so the test does not depend on the multiprocessing
    start method.
    """
    pytest.importorskip("cv2")

    _, data_model = temp_data_dir

    # Map impl_class to ocr_type string expected by OCRProcessingStage
    impl_to_type = {
        EasyOCRImpl: "easyocr",
        OLMOcrImpl: "olmocr",
        OcrMyPDFImpl: "ocrmypdf",
    }

    stage = OCRProcessingStage(
        data_model=data_model,
        ocr_type=impl_to_type[impl_class],
        batch_size=1,
        max_workers=2,
        **init_args,
    )

    # Create sample images on disk that cv2 can read. Two pages with batch_size=1
    # means more than one batch, so page-to-text alignment is exercised too.
    import cv2

    digest = "abc123def456abc123def456abc123def45"
    img_dir = os.path.join(data_model.image_directory, digest)
    os.makedirs(img_dir, exist_ok=True)
    for page_num in range(2):
        img = _create_test_image(f"PIPELINE TEST {page_num}")
        img_path = os.path.join(img_dir, f"{digest}_{page_num}.jpeg")
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    _InlineProcessPoolExecutor.map_calls = 0
    with (
        patch(
            "govscape.processing.ocr_processing_stage._build_ocr_engine",
            return_value=_StaticTextOCR(),
        ),
        patch(
            "govscape.processing.ocr_processing_stage.ProcessPoolExecutor",
            _InlineProcessPoolExecutor,
        ),
    ):
        stage.run()

    assert _InlineProcessPoolExecutor.map_calls == 1
    for page_num in range(2):
        with open(data_model.txt_page_path(digest, page_num), encoding="utf-8") as f:
            assert f.read() == _StaticTextOCR.TEXT


def test_ocr_processing_stage_skips_pages_with_text_or_blank(temp_data_dir):
    cv2 = pytest.importorskip("cv2")
    _, data_model = temp_data_dir

    digest = "abc123def456abc123def456abc123def45"
    img_dir = os.path.join(data_model.image_directory, digest)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(data_model.txt_pdf_directory(digest), exist_ok=True)
    text_layer = "A page with a real text layer from the PDF."
    pages = {
        0: (_create_test_image("PAGE ZERO"), text_layer),
        1: (_create_test_image("PAGE ONE"), "12"),
        2: (np.full((80, 300, 3), 255, dtype=np.uint8), None),
        3: (_create_test_image("PAGE THREE"), None),
    }
    for page_num, (img, txt) in pages.items():
        cv2.imwrite(os.path.join(img_dir, f"{digest}_{page_num}.jpeg"), img)
        if txt is not None:
            with open(data_model.txt_page_path(digest, page_num), "w") as f:
                f.write(txt)

    stage = OCRProcessingStage(data_model=data_model, ocr_type="easyocr")
    with (
        patch(
            "govscape.processing.ocr_processing_stage._build_ocr_engine",
            return_value=_StaticTextOCR(),
        ),
        patch(
            "govscape.processing.ocr_processing_stage.ProcessPoolExecutor",
            _InlineProcessPoolExecutor,
        ),
    ):
        stage.run()

    def page_text(page_num):
        path = data_model.txt_page_path(digest, page_num)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return f.read()

    assert page_text(0) == text_layer
    assert page_text(1) == _StaticTextOCR.TEXT
    assert page_text(2) is None
    assert page_text(3) == _StaticTextOCR.TEXT
