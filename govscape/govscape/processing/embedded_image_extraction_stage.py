import io
import logging
import os
import shutil
from multiprocessing import get_context
from pathlib import Path

import fitz
from PIL import Image, ImageFile

from .processing_stage import ProcessingStage

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _extract_single_pdf(
    pdf_directory, extracted_img_path, embeddings_img_e_path, pdf_path
):
    full_pdf_path = Path(pdf_directory) / Path(pdf_path)
    output_img_dir_path = Path(extracted_img_path) / Path(pdf_path).stem
    output_img_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        with fitz.open(full_pdf_path) as pdf_doc:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            empty = True
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                for i, img in enumerate(page.get_images(full=True)):
                    if i == 4:
                        break
                    empty = False

                    xref = img[0]
                    image_dict = pdf_doc.extract_image(xref)
                    image_bytes = image_dict["image"]

                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        image.load()
                    except Exception:
                        continue

                    image_path = (
                        Path(output_img_dir_path) / f"{pdf_name}_{page_num}_{i}.jpeg"
                    )
                    image = image.convert("RGB")

                    if (
                        image.size[0] < 80
                        or image.size[1] < 80
                        or image.size[0] > 7000
                        or image.size[1] > 7000
                    ):
                        continue
                    image.save(image_path, "JPEG")

            if empty:
                shutil.rmtree(output_img_dir_path)

    except Exception as e:
        logging.error(f"can't open PDF {pdf_path}: {e}")


class EmbeddedImageExtractionStage(ProcessingStage):
    def __init__(
        self, pdfs_path, extracted_img_path, embeddings_img_e_path, pdf_files, cpu_count
    ):
        self.pdfs_path = pdfs_path
        self.extracted_img_path = extracted_img_path
        self.embeddings_img_e_path = embeddings_img_e_path
        self.pdf_files = pdf_files
        self.cpu_count = cpu_count

    def validate(self) -> list[str]:
        errors = []
        if not os.path.isdir(self.pdfs_path):
            errors.append(f"PDFs input directory does not exist: {self.pdfs_path}")
        return errors

    def run(self):
        ctx = get_context("spawn")
        with ctx.Pool(processes=self.cpu_count) as pool:
            pool.starmap(
                _extract_single_pdf,
                [
                    (
                        self.pdfs_path,
                        self.extracted_img_path,
                        self.embeddings_img_e_path,
                        file,
                    )
                    for file in self.pdf_files
                ],
            )
