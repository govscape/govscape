import json
import os
import shutil
from multiprocessing import get_context

import pypdfium2

from ..config import DataModel
from .processing_stage import ProcessingStage


def _convert_single_pdf(data_model, pdf_file):
    pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
    os.makedirs(data_model.txt_pdf_directory(pdf_name), exist_ok=True)
    os.makedirs(data_model.img_pdf_directory(pdf_name), exist_ok=True)
    try:
        pdf = pypdfium2.PdfDocument(pdf_file)
        num_pages = len(pdf)
        pretty_name = pdf.get_metadata_value("Title").strip()
        json_data = {}
        creation_date = pdf.get_metadata_value("CreationDate")
        if len(pretty_name) == 0:
            pretty_name = ""
        if len(creation_date) == 0:
            creation_date = "Unknown"
        json_data["pretty_name"] = pretty_name
        json_data["digest"] = pdf_name
        json_data["creation_date"] = creation_date
        json_data["num_pages"] = num_pages
        os.makedirs(data_model.metadata_pdf_directory(pdf_name), exist_ok=True)
        with open(data_model.metadata_file_path(pdf_name), "w") as json_file:
            json.dump(json_data, json_file, indent=4)
    except Exception:
        return False

    # Write each page as soon as it is rendered: holding every rendered page of a
    # long PDF (or of a large-format one) in memory can use gigabytes.
    for page_num in range(num_pages):
        try:
            page = pdf[page_num]
            page_text = page.get_textpage().get_text_bounded()
            image = page.render(scale=1.0).to_pil()
        except Exception:
            # Leave no partial pages from a PDF that failed part way through.
            for directory in (
                data_model.txt_pdf_directory(pdf_name),
                data_model.img_pdf_directory(pdf_name),
            ):
                shutil.rmtree(directory)
                os.makedirs(directory)
            return False
        if page_text:
            with open(
                data_model.txt_page_path(pdf_name, page_num), "w", encoding="utf-8"
            ) as text_file:
                text_file.write(page_text)
        image.save(data_model.img_page_path(pdf_name, page_num), format="JPEG")
    return True


class PDFExtractionStage(ProcessingStage):
    def __init__(self, data_model: DataModel, pdf_files: list[str], cpu_count: int):
        self.data_model = data_model
        self.pdf_files = pdf_files
        self.cpu_count = cpu_count

    def validate(self) -> None:
        missing = [f for f in self.pdf_files if not os.path.isfile(f)]
        if missing:
            raise ValueError(
                f"{len(missing)} PDF file(s) not found, e.g.: {missing[0]}"
            )

    def run(self):
        ctx = get_context("spawn")
        with ctx.Pool(processes=self.cpu_count) as pool:
            results = pool.starmap(
                _convert_single_pdf,
                [(self.data_model, pdf_file) for pdf_file in self.pdf_files],
            )
            # Let workers exit on their own: Pool.__exit__ terminates them, and some
            # native libs' SIGTERM handlers can deadlock.
            pool.close()
            pool.join()
        return sum(results)
