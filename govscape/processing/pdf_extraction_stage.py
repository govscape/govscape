import json
import os
import shutil
from multiprocessing import get_context

import numpy as np

import pypdfium2
from PIL import Image

from ..config import DataModel
from .processing_stage import ProcessingStage

# pdfium renders at 72 dpi (one pixel per PDF point) at scale 1.
PDF_POINTS_PER_INCH = 72
# Tesseract is most accurate on text rendered at about 300 dpi.
OCR_DPI = 300
# A page needs OCR if its PDF text layer is missing or nearly empty (e.g. a bare
# page number)...
MAX_TEXT_LAYER_CHARS = 20
# ...and it isn't blank: at least MIN_INK_FRACTION of its pixels are darker than
# INK_LEVEL. On a sample of the archive's pages without a text layer, this
# excluded only truly blank pages (~9% of them).
INK_LEVEL = 160
MIN_INK_FRACTION = 0.002
# Tesseract's time grows much faster than a page's pixels: a 43x35 in drawing at
# 300 dpi (114 MP) takes ~160 s against ~1 s for a letter page (8 MP). Larger
# pages are rendered for OCR at a lower dpi, to about this many pixels.
MAX_OCR_PIXELS = 40_000_000
# Long PDFs are extracted in tasks of this many pages, so one 10,000-page PDF
# doesn't keep a single process busy while the others sit idle.
PAGES_PER_TASK = 100


def needs_ocr(page_text: str, page_image: Image.Image) -> bool:
    if len(page_text.strip()) > MAX_TEXT_LAYER_CHARS:
        return False
    gray = np.asarray(page_image.convert("L"))
    return bool((gray < INK_LEVEL).mean() >= MIN_INK_FRACTION)


def _remove_partial_pages(data_model: DataModel, pdf_name: str) -> None:
    # Keep the (now empty) txt and img directories that every parsed PDF has.
    for directory in (
        data_model.txt_pdf_directory(pdf_name),
        data_model.img_pdf_directory(pdf_name),
    ):
        shutil.rmtree(directory)
        os.makedirs(directory)
    shutil.rmtree(data_model.ocr_img_pdf_directory(pdf_name), ignore_errors=True)


def _ocr_scale(page) -> float:
    """pdfium render scale for OCR: OCR_DPI, lowered for large-format pages
    (drawings, maps) so the render stays within MAX_OCR_PIXELS."""
    width, height = page.get_size()
    full_size_scale = OCR_DPI / PDF_POINTS_PER_INCH
    return min(full_size_scale, (MAX_OCR_PIXELS / (width * height)) ** 0.5)


def _page_count(pdf_file: str) -> int | None:
    try:
        return len(pypdfium2.PdfDocument(pdf_file))
    except Exception:
        return None


def _write_metadata(data_model: DataModel, pdf_name: str, pdf, num_pages: int):
    pretty_name = pdf.get_metadata_value("Title").strip()
    creation_date = pdf.get_metadata_value("CreationDate") or "Unknown"
    os.makedirs(data_model.metadata_pdf_directory(pdf_name), exist_ok=True)
    with open(data_model.metadata_file_path(pdf_name), "w") as json_file:
        json.dump(
            {
                "pretty_name": pretty_name,
                "digest": pdf_name,
                "creation_date": creation_date,
                "num_pages": num_pages,
            },
            json_file,
            indent=4,
        )


def _convert_pages(data_model, pdf_file, first_page, end_page, render_ocr_pages):
    """Extract pages [first_page, end_page) of pdf_file (the whole PDF if
    end_page is None), and the PDF's metadata with its first page. Returns
    whether they could all be parsed; on failure the caller removes the PDF's
    partial pages."""
    pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
    os.makedirs(data_model.txt_pdf_directory(pdf_name), exist_ok=True)
    os.makedirs(data_model.img_pdf_directory(pdf_name), exist_ok=True)
    try:
        pdf = pypdfium2.PdfDocument(pdf_file)
        num_pages = len(pdf)
        if first_page == 0:
            _write_metadata(data_model, pdf_name, pdf, num_pages)
    except Exception:
        return False

    # Write each page as soon as it is rendered: holding every rendered page of a
    # long PDF (or of a large-format one) in memory can use gigabytes.
    for page_num in range(first_page, num_pages if end_page is None else end_page):
        try:
            page = pdf[page_num]
            page_text = page.get_textpage().get_text_bounded()
            image = page.render(scale=1.0).to_pil()
            ocr_image = None
            if render_ocr_pages and needs_ocr(page_text, image):
                ocr_image = page.render(scale=_ocr_scale(page), grayscale=True)
                ocr_image = ocr_image.to_pil()
        except Exception:
            return False
        # Errors while writing (e.g. a full disk) aren't the PDF's fault, so they
        # propagate and fail the batch instead.
        if page_text:
            with open(
                data_model.txt_page_path(pdf_name, page_num), "w", encoding="utf-8"
            ) as text_file:
                text_file.write(page_text)
        image.save(data_model.img_page_path(pdf_name, page_num), format="JPEG")
        if ocr_image is not None:
            ocr_path = data_model.ocr_img_page_path(pdf_name, page_num)
            os.makedirs(os.path.dirname(ocr_path), exist_ok=True)
            # PNG: JPEG artifacts around glyphs cost OCR accuracy. The render is
            # read once and deleted, so favour fast compression over small files.
            ocr_image.save(ocr_path, format="PNG", compress_level=1)
    return True


def _convert_single_pdf(data_model, pdf_file, render_ocr_pages=False):
    if _convert_pages(data_model, pdf_file, 0, None, render_ocr_pages):
        return True
    _remove_partial_pages(data_model, os.path.splitext(os.path.basename(pdf_file))[0])
    return False


class PDFExtractionStage(ProcessingStage):
    """Extracts each PDF's metadata, and each page's text and 72 dpi image.

    With render_ocr_pages, the pages that need OCR are also rendered at OCR_DPI
    (lower for large-format pages) into data_model.ocr_image_directory, for
    OCRProcessingStage. Long PDFs are split into tasks of PAGES_PER_TASK pages,
    longest PDFs first, so the processes finish together.
    """

    def __init__(
        self,
        data_model: DataModel,
        pdf_files: list[str],
        cpu_count: int,
        render_ocr_pages: bool = False,
    ):
        self.data_model = data_model
        self.pdf_files = pdf_files
        self.cpu_count = cpu_count
        self.render_ocr_pages = render_ocr_pages

    def validate(self) -> None:
        missing = [f for f in self.pdf_files if not os.path.isfile(f)]
        if missing:
            raise ValueError(
                f"{len(missing)} PDF file(s) not found, e.g.: {missing[0]}"
            )

    def run(self):
        tasks = []  # (pdf_file, first_page, end_page), longest PDFs first
        for pdf_file, num_pages in sorted(
            ((f, _page_count(f)) for f in self.pdf_files),
            key=lambda pdf: -(pdf[1] or 0),
        ):
            if num_pages is None:  # Unreadable; its task reports the failure.
                tasks.append((pdf_file, 0, None))
                continue
            for first_page in range(0, max(num_pages, 1), PAGES_PER_TASK):
                end_page = min(first_page + PAGES_PER_TASK, num_pages)
                tasks.append((pdf_file, first_page, end_page))

        ctx = get_context("spawn")
        with ctx.Pool(processes=self.cpu_count) as pool:
            results = pool.starmap(
                _convert_pages,
                [
                    (self.data_model, pdf_file, first, end, self.render_ocr_pages)
                    for pdf_file, first, end in tasks
                ],
                chunksize=1,
            )
            # Let workers exit on their own: Pool.__exit__ terminates them, and some
            # native libs' SIGTERM handlers can deadlock.
            pool.close()
            pool.join()

        failed = {
            pdf_file
            for (pdf_file, _, _), ok in zip(tasks, results, strict=True)
            if not ok
        }
        for pdf_file in failed:
            # Leave no partial pages from a PDF that failed part way through.
            pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
            _remove_partial_pages(self.data_model, pdf_name)
        return len(self.pdf_files) - len(failed)
