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


def extracted_pdfs(data_model: DataModel) -> list[tuple[str, int]]:
    """(digest, page count) of each PDF fully extracted into data_model, in
    digest order."""
    if not os.path.isdir(data_model.metadata_directory):
        return []
    pdfs = []
    for digest in sorted(os.listdir(data_model.metadata_directory)):
        with open(data_model.metadata_file_path(digest)) as f:
            pdfs.append((digest, json.load(f)["num_pages"]))
    return pdfs


def _pdf_name(pdf_file: str) -> str:
    return os.path.splitext(os.path.basename(pdf_file))[0]


def _remove_pdf(data_model: DataModel, pdf_name: str) -> None:
    """Remove everything extracted from a PDF that couldn't be fully extracted,
    so that it isn't one of the extracted_pdfs."""
    for directory in data_model.extracted_pdf_directories(pdf_name):
        shutil.rmtree(directory, ignore_errors=True)


def _page_count(pdf_file: str) -> int | None:
    try:
        return len(pypdfium2.PdfDocument(pdf_file))
    except Exception:
        return None


def _convert_pages(
    data_model: DataModel,
    pdf_file: str,
    first_page: int,
    end_page: int,
    render_ocr_pages: bool,
) -> bool:
    """Extract pages [first_page, end_page) of pdf_file; the task that starts at
    page 0 also writes the PDF's metadata.json. Returns whether they could all be
    parsed; on failure the caller removes the PDF."""
    pdf_name = _pdf_name(pdf_file)
    os.makedirs(data_model.txt_pdf_directory(pdf_name), exist_ok=True)
    os.makedirs(data_model.img_pdf_directory(pdf_name), exist_ok=True)
    try:
        pdf = pypdfium2.PdfDocument(pdf_file)
        num_pages = len(pdf)
        if first_page == 0:
            metadata = {
                "pretty_name": pdf.get_metadata_value("Title").strip(),
                "digest": pdf_name,
                "creation_date": pdf.get_metadata_value("CreationDate") or "Unknown",
                "num_pages": num_pages,
            }
            os.makedirs(data_model.metadata_pdf_directory(pdf_name), exist_ok=True)
            with open(data_model.metadata_file_path(pdf_name), "w") as json_file:
                json.dump(metadata, json_file, indent=4)
    except Exception:
        return False

    # Write each page as soon as it is rendered: holding every rendered page of a
    # long PDF (or of a large-format one) in memory can use gigabytes.
    for page_num in range(first_page, end_page):
        try:
            page = pdf[page_num]
            page_text = page.get_textpage().get_text_bounded()
            image = page.render(scale=1.0).to_pil()
            ocr_image = None
            if render_ocr_pages and needs_ocr(page_text, image):
                # At OCR_DPI, lowered for large-format pages (drawings, maps) so
                # the render stays within MAX_OCR_PIXELS.
                width, height = page.get_size()
                scale = min(
                    OCR_DPI / PDF_POINTS_PER_INCH,
                    (MAX_OCR_PIXELS / (width * height)) ** 0.5,
                )
                ocr_image = page.render(scale=scale, grayscale=True).to_pil()
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
    num_pages = _page_count(pdf_file) or 0
    if _convert_pages(data_model, pdf_file, 0, num_pages, render_ocr_pages):
        return True
    _remove_pdf(data_model, _pdf_name(pdf_file))
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
        tasks: list[tuple[str, int, int]] = []  # Longest PDFs first.
        for pdf_file, num_pages in sorted(
            ((f, _page_count(f)) for f in self.pdf_files),
            key=lambda pdf: -(pdf[1] or 0),
        ):
            if num_pages is None:  # Unreadable; its task reports the failure.
                tasks.append((pdf_file, 0, 0))
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
            _remove_pdf(self.data_model, _pdf_name(pdf_file))
        return len(self.pdf_files) - len(failed)
