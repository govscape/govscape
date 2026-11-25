"""Tests for OCR implementations and OCRProcessingStage."""

import os
import shutil

import pytest

import numpy as np

from PIL import Image, ImageDraw, ImageFont

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
)


def _create_test_image(text: str, size=(300, 80)) -> np.ndarray:
    """A white RGB image with black text."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    ImageDraw.Draw(img).text(
        (10, 10), text, fill=(0, 0, 0), font=ImageFont.load_default()
    )
    return np.array(img)


def _scanned_page() -> Image.Image:
    """A letter-size page image covered in lines of text, like a scan."""
    page = Image.new("RGB", (612, 792), color=(255, 255, 255))
    draw = ImageDraw.Draw(page)
    for line in range(40):
        draw.text((40, 30 + 18 * line), "SCANNED REPORT TEXT " * 4, fill=(0, 0, 0))
    return page


OCR_IMPLS = [
    (EasyOCRImpl, {"languages": ["en"], "gpu": False}, "easyocr"),
    (OLMOcrImpl, {"model_name": "default"}, "olmocr"),
    (OcrMyPDFImpl, {"language": "eng"}, "pytesseract"),
]


@pytest.mark.parametrize("impl_class,init_args,skip_pkg", OCR_IMPLS)
def test_ocr_implementations_on_sample_images(impl_class, init_args, skip_pkg):
    """Each OCR implementation extracts the expected text; skipped for those whose
    dependencies are not installed."""
    pytest.importorskip(skip_pkg)
    # olmOCR runs a ~7B vision-language model (large download, slow, GPU-bound),
    # so its real-inference test is opt-in rather than run on every invocation.
    if impl_class is OLMOcrImpl and not os.environ.get("RUN_OLMOCR_MODEL"):
        pytest.skip("Set RUN_OLMOCR_MODEL=1 to run the olmOCR model test.")

    expected = ["TEST ONE", "HELLO 123"]
    ocr = impl_class(**init_args)
    try:
        ocr.validate()
    except ImportError:
        pytest.skip("Required OCR dependency is not available")

    extracted = ocr.extract_text([_create_test_image(text) for text in expected])
    assert len(extracted) == len(expected)
    for text, found in zip(expected, extracted, strict=True):
        assert isinstance(found, str)
        assert text.lower().split()[0] in found.lower()


def test_extraction_renders_only_pages_that_need_ocr(tmp_path):
    data_model = DataModel(tmp_path.as_posix())
    # PIL writes image-only PDFs (no text layer), one page per image, at 72 dpi:
    # a blank page, a scanned letter page and a 43 x 35 in drawing (which would
    # be ~135 MP at 300 dpi).
    blank = Image.new("RGB", (612, 792), color=(255, 255, 255))
    drawing = Image.new("RGB", (43 * 72, 35 * 72), color=(255, 255, 255))
    ImageDraw.Draw(drawing).rectangle((100, 100, 2000, 1500), fill=(0, 0, 0))
    digest = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    pdf_path = (tmp_path / f"{digest}.pdf").as_posix()
    blank.save(
        pdf_path, format="PDF", save_all=True, append_images=[_scanned_page(), drawing]
    )

    _convert_single_pdf(data_model, pdf_path, render_ocr_pages=True)

    for page_num in range(3):
        assert os.path.exists(data_model.img_page_path(digest, page_num))
    assert not os.path.exists(data_model.ocr_img_page_path(digest, 0))
    with Image.open(data_model.ocr_img_page_path(digest, 1)) as render:
        assert render.mode == "L"
        assert render.size == pytest.approx(
            (612 * OCR_DPI / 72, 792 * OCR_DPI / 72), abs=1
        )
    with Image.open(data_model.ocr_img_page_path(digest, 2)) as render:
        width, height = render.size
    # Large-format pages are rendered at a lower dpi; pdfium rounds each side up.
    assert 0.99 * MAX_OCR_PIXELS < width * height < 1.01 * MAX_OCR_PIXELS
    assert abs(width / height - 43 / 35) < 0.01


def test_scanned_pdf_is_extracted_and_ocred(tmp_path):
    pytest.importorskip("pytesseract")
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed")
    data_model = DataModel(tmp_path.as_posix())
    digest = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    pdf_path = (tmp_path / f"{digest}.pdf").as_posix()
    _scanned_page().save(pdf_path, format="PDF")
    _convert_single_pdf(data_model, pdf_path, render_ocr_pages=True)

    OCRProcessingStage(data_model, ocr_type="ocrmypdf", num_processes=1).run()

    with open(data_model.txt_page_path(digest, 0), encoding="utf-8") as f:
        assert "SCANNED REPORT TEXT" in f.read()
    assert not os.path.exists(data_model.ocr_image_directory)


def test_tesseract_is_stopped_at_its_time_limit(monkeypatch):
    pytest.importorskip("pytesseract")
    if shutil.which("tesseract") is None:
        pytest.skip("tesseract binary not installed")
    monkeypatch.setattr(OcrMyPDFImpl, "TIMEOUT_S", 0.01)
    monkeypatch.setattr(OcrMyPDFImpl, "RETRY_TIMEOUT_S", 0.01)
    page = _create_test_image("SLOW PAGE " * 20, (2400, 3000))
    assert OcrMyPDFImpl(max_workers=1).extract_text([page]) == [None]
