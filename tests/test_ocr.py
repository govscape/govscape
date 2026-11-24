"""Tests for OCR implementations and OCRProcessingStage."""

import os
import tempfile
from unittest.mock import patch

import pytest

import numpy as np

from PIL import Image, ImageDraw

from govscape.config import DataModel
from govscape.processing import OCRProcessingStage
from govscape.processing.ocr import (
    EasyOCRImpl,
    OcrMyPDFImpl,
    OLMOcrImpl,
)
from govscape.processing.pdf_extraction_stage import (
    MAX_OCR_PIXELS,
    OCR_DPI,
    _convert_single_pdf,
    needs_ocr,
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


def _write_ocr_render(data_model, digest, page_num, image):
    path = data_model.ocr_img_page_path(digest, page_num)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image).convert("L").save(path, format="PNG")


@pytest.mark.parametrize("impl_class,init_args,skip_pkg", OCR_IMPLS)
def test_ocr_processing_stage_writes_txt(
    impl_class, init_args, skip_pkg, temp_data_dir
):
    """OCRProcessingStage should dispatch batches through a ProcessPoolExecutor,
    write one txt file per rendered page, and delete the renders, for every OCR
    engine type.

    The worker-side engine is replaced with a fake so no real OCR model is loaded,
    and the pool runs inline so the test does not depend on the multiprocessing
    start method.
    """
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
        pages_per_batch=1,
        num_processes=2,
        **init_args,
    )

    # Two pages with batch_size=1 means more than one batch, so page-to-text
    # alignment is exercised too.
    digest = "abc123def456abc123def456abc123def45"
    for page_num in range(2):
        _write_ocr_render(
            data_model,
            digest,
            page_num,
            _create_test_image(f"PIPELINE TEST {page_num}"),
        )
    os.makedirs(data_model.txt_pdf_directory(digest), exist_ok=True)
    with open(data_model.txt_page_path(digest, 1), "w") as f:
        f.write("12")

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
    assert not os.path.exists(data_model.ocr_image_directory)


def test_ocr_processing_stage_with_nothing_to_ocr(temp_data_dir):
    _, data_model = temp_data_dir
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
    assert not os.path.exists(data_model.txt_directory) or not os.listdir(
        data_model.txt_directory
    )


def _scanned_page() -> Image.Image:
    """A letter-size page image covered in lines of text, like a scan."""
    from PIL import ImageDraw

    page = Image.new("RGB", (612, 792), color=(255, 255, 255))
    draw = ImageDraw.Draw(page)
    for line in range(40):
        draw.text((40, 30 + 18 * line), "SCANNED REPORT TEXT " * 4, fill=(0, 0, 0))
    return page


def test_needs_ocr_skips_pages_with_a_text_layer_or_blank():
    page = Image.fromarray(_create_test_image("SCANNED TEXT"))
    blank = Image.new("RGB", (300, 80), color=(255, 255, 255))
    assert needs_ocr("", page)
    assert needs_ocr("  12 \n", page)  # a bare page number isn't a text layer
    assert not needs_ocr("A page with a real text layer from the PDF.", page)
    assert not needs_ocr("", blank)


def test_extraction_renders_only_pages_that_need_ocr_at_ocr_dpi(temp_data_dir):
    tmpdir, data_model = temp_data_dir
    # PIL writes image-only PDFs (no text layer), one page per image, at 72 dpi.
    scanned = _scanned_page()
    blank = Image.new("RGB", (612, 792), color=(255, 255, 255))
    digest = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    pdf_path = os.path.join(tmpdir, f"{digest}.pdf")
    blank.save(pdf_path, format="PDF", save_all=True, append_images=[scanned])

    _convert_single_pdf(data_model, pdf_path, render_ocr_pages=True)

    assert os.path.exists(data_model.img_page_path(digest, 0))
    assert os.path.exists(data_model.img_page_path(digest, 1))
    assert not os.path.exists(data_model.ocr_img_page_path(digest, 0))
    with Image.open(data_model.ocr_img_page_path(digest, 1)) as render:
        assert render.mode == "L"
        width, height = render.size
        assert abs(width - 612 * OCR_DPI / 72) <= 1
        assert abs(height - 792 * OCR_DPI / 72) <= 1
    with Image.open(data_model.img_page_path(digest, 1)) as page_image:
        assert page_image.size == (612, 792)


def test_extraction_without_render_ocr_pages_renders_nothing_for_ocr(temp_data_dir):
    tmpdir, data_model = temp_data_dir
    scanned = _scanned_page()
    digest = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    pdf_path = os.path.join(tmpdir, f"{digest}.pdf")
    scanned.save(pdf_path, format="PDF")

    _convert_single_pdf(data_model, pdf_path)

    assert os.path.exists(data_model.img_page_path(digest, 0))
    assert not os.path.exists(data_model.ocr_image_directory)


def test_an_unreadable_render_fails_only_its_own_page(temp_data_dir):
    _, data_model = temp_data_dir
    digest = "abc123def456abc123def456abc123def45"
    _write_ocr_render(data_model, digest, 0, _create_test_image("GOOD PAGE"))
    with open(data_model.ocr_img_page_path(digest, 1), "wb") as f:
        f.write(b"not a png")

    stage = OCRProcessingStage(data_model=data_model, ocr_type="ocrmypdf")
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

    with open(data_model.txt_page_path(digest, 0), encoding="utf-8") as f:
        assert f.read() == _StaticTextOCR.TEXT
    assert not os.path.exists(data_model.txt_page_path(digest, 1))


def test_ocr_renders_large_format_pages_at_a_lower_dpi(temp_data_dir):
    tmpdir, data_model = temp_data_dir
    # A 43 x 35 in drawing would be ~135 MP at 300 dpi.
    drawing = Image.new("RGB", (43 * 72, 35 * 72), color=(255, 255, 255))
    ImageDraw.Draw(drawing).rectangle((100, 100, 2000, 1500), fill=(0, 0, 0))
    digest = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    pdf_path = os.path.join(tmpdir, f"{digest}.pdf")
    drawing.save(pdf_path, format="PDF", resolution=72)

    _convert_single_pdf(data_model, pdf_path, render_ocr_pages=True)

    with Image.open(data_model.ocr_img_page_path(digest, 0)) as render:
        width, height = render.size
    # pdfium rounds each side up to whole pixels.
    assert 0.99 * MAX_OCR_PIXELS < width * height < 1.01 * MAX_OCR_PIXELS
    assert abs(width / height - 43 / 35) < 0.01


def _timeout_then(results):
    """A fake pytesseract.image_to_string: each call returns the next of results,
    or raises Tesseract's timeout error for None."""
    calls = []

    def image_to_string(image, lang, timeout):
        calls.append((image.size, timeout))
        result = results[len(calls) - 1]
        if result is None:
            raise RuntimeError("Tesseract process timeout")
        return result

    return image_to_string, calls


def test_tesseract_retries_a_slow_page_at_half_resolution(monkeypatch):
    pytest.importorskip("pytesseract")
    fake, calls = _timeout_then([None, "text"])
    monkeypatch.setattr(
        "govscape.processing.ocr.ocrmypdf_impl.pytesseract.image_to_string", fake
    )
    ocr = OcrMyPDFImpl(max_workers=1)
    assert ocr.extract_text([np.zeros((100, 80), np.uint8)]) == ["text"]
    assert calls == [((80, 100), OcrMyPDFImpl.TIMEOUT_S), ((40, 50), 30)]


def test_tesseract_gives_up_on_a_page_that_times_out_twice(monkeypatch):
    pytest.importorskip("pytesseract")
    fake, calls = _timeout_then([None, None])
    monkeypatch.setattr(
        "govscape.processing.ocr.ocrmypdf_impl.pytesseract.image_to_string", fake
    )
    ocr = OcrMyPDFImpl(max_workers=1)
    assert ocr.extract_text([np.zeros((100, 80), np.uint8)]) == [None]
    assert len(calls) == 2


def test_tesseract_is_stopped_at_its_time_limit(monkeypatch):
    pytest.importorskip("pytesseract")
    import shutil as _shutil

    if _shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed")
    monkeypatch.setattr(OcrMyPDFImpl, "TIMEOUT_S", 0.01)
    monkeypatch.setattr(OcrMyPDFImpl, "RETRY_TIMEOUT_S", 0.01)
    page = np.array(
        Image.fromarray(_create_test_image("SLOW PAGE " * 20, (2400, 3000)))
    )
    assert OcrMyPDFImpl(max_workers=1).extract_text([page]) == [None]
