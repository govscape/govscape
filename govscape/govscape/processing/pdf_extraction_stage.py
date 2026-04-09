import json
import os
from multiprocessing import get_context

import pypdfium2

from .processing_stage import ProcessingStage


def _convert_single_pdf(txts_path, imgs_path, pdfs_path, metadata_dir, pdf_file):
    pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
    pdf_path = os.path.join(pdfs_path, pdf_file)
    pdf_txt_subdir = os.path.join(txts_path, pdf_name)
    pdf_img_subdir = os.path.join(imgs_path, pdf_name)
    os.makedirs(pdf_txt_subdir, exist_ok=True)
    os.makedirs(pdf_img_subdir, exist_ok=True)
    try:
        pdf = pypdfium2.PdfDocument(pdf_path)
        num_pages = len(pdf)
        gov_name = pdf.get_metadata_value("Title")

        json_data = {}
        timestamp = pdf.get_metadata_value("CreationDate")
        if len(gov_name) == 0:
            gov_name = "Unknown"
        if len(timestamp) == 0:
            timestamp = "Unknown"
        json_data["gov_name"] = gov_name
        json_data["timestamp"] = timestamp
        json_data["num_pages"] = num_pages
        pdf_metadata_dir = os.path.join(
            metadata_dir, os.path.splitext(os.path.basename(pdf_file))[0]
        )
        os.makedirs(pdf_metadata_dir, exist_ok=True)
        json_file_path = os.path.join(pdf_metadata_dir, "metadata.json")
        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        text = []
        images = []
        for i in range(num_pages):
            page = pdf[i]
            page_text = page.get_textpage().get_text_bounded()
            text.append(page_text)
            pil_image = page.render(scale=1.0).to_pil()
            images.append(pil_image)
    except Exception:
        return False

    for page_num, page_text in enumerate(text):
        txt_file_path = os.path.join(pdf_txt_subdir, f"{pdf_name}_{page_num}.txt")
        if page_text and len(page_text) != 0:
            with open(txt_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(page_text)

        img_file_path = os.path.join(pdf_img_subdir, f"{pdf_name}_{page_num}.jpeg")
        image = images[page_num]
        image.save(img_file_path, format="JPEG")
    return True


class PDFExtractionStage(ProcessingStage):
    def __init__(
        self, pdfs_path, txts_path, img_path, metadata_dir, pdf_files, cpu_count
    ):
        self.pdfs_path = pdfs_path
        self.txts_path = txts_path
        self.img_path = img_path
        self.metadata_dir = metadata_dir
        self.pdf_files = pdf_files
        self.cpu_count = cpu_count

    def validate(self) -> list[str]:
        errors = []
        if not os.path.isdir(self.pdfs_path):
            errors.append(f"PDFs input directory does not exist: {self.pdfs_path}")
        return errors

    def run(self):
        os.makedirs(self.txts_path, exist_ok=True)
        ctx = get_context("forkserver")
        with ctx.Pool(processes=self.cpu_count) as pool:
            results = pool.starmap(
                _convert_single_pdf,
                [
                    (
                        self.txts_path,
                        self.img_path,
                        self.pdfs_path,
                        self.metadata_dir,
                        file,
                    )
                    for file in self.pdf_files
                ],
            )
        return sum(results)
