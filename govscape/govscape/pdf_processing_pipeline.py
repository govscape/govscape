import logging
import os
import time

from .config import DataModel
from .processing import (
    OCRProcessingStage,
    PageImageEmbeddingStage,
    PDFExtractionStage,
    TextEmbeddingStage,
)


class PDFProcessingPipeline:
    def __init__(
        self,
        data_dir: str,
        text_model_type: str,
        visual_model_type: str,
        ocr_type: str = None,
        **ocr_kwargs,
    ):
        self.data_model = DataModel(data_dir)
        self.cpu_count = os.cpu_count() or 1
        self.text_model_type = text_model_type
        self.visual_model_type = visual_model_type
        self.ocr_type = ocr_type
        self.ocr_kwargs = ocr_kwargs

    def process_pdfs(
        self,
        pdf_files: list[str],
        do_text_embedding: bool,
        do_img_embedding: bool,
        do_metadata_collection: bool,
        do_ocr: bool = False,
    ):
        run_extraction = (
            do_text_embedding or do_img_embedding or do_metadata_collection or do_ocr
        )

        time1 = time.time()
        pdfs_successfully_parsed = 0
        if run_extraction:
            logging.info("Converting pdfs to txts and page images")
            pdf_extraction_stage = PDFExtractionStage(
                data_model=self.data_model,
                pdf_files=pdf_files,
                cpu_count=self.cpu_count,
            )
            pdf_extraction_stage.validate()
            pdfs_successfully_parsed = pdf_extraction_stage.run()
        parsed_summary = (
            f"PDFs successfully parsed: {pdfs_successfully_parsed} / {len(pdf_files)}"
        )
        logging.info(parsed_summary)

        time2 = time.time()
        if do_ocr and self.ocr_type:
            logging.info(f"Running OCR extraction using {self.ocr_type}")
            ocr_stage = OCRProcessingStage(
                data_model=self.data_model,
                ocr_type=self.ocr_type,
                **self.ocr_kwargs,
            )
            ocr_stage.validate()
            ocr_stage.run()
        time3 = time.time()

        if do_text_embedding:
            logging.info("Converting txts to embeddings")
            text_embedding_stage = TextEmbeddingStage(
                data_model=self.data_model,
                model_type=self.text_model_type,
            )
            text_embedding_stage.validate()
            text_embedding_stage.run()
        time4 = time.time()

        if do_img_embedding:
            logging.info("Converting imgs to embeddings")
            page_image_embedding_stage = PageImageEmbeddingStage(
                data_model=self.data_model,
                model_type=self.visual_model_type,
                cpu_count=self.cpu_count,
            )
            page_image_embedding_stage.validate()
            page_image_embedding_stage.run()
        time5 = time.time()

        pdf_to_txt_img_metadata = time2 - time1
        # Compute ocr_time consistently as the interval between time2 and time3.
        # If OCR was skipped, time3 == time2 so this will be ~0.0.
        ocr_time = time3 - time2
        text_embed_time = time4 - time3
        img_embed_time = time5 - time4

        logging.info(f"pdf -> txt, img, metadata time: {pdf_to_txt_img_metadata}")
        if do_ocr:
            logging.info(f"ocr processing time: {ocr_time}")
        logging.info(f"txt -> embed time: {text_embed_time}")
        logging.info(f"img per page -> embed time: {img_embed_time}")

        return (
            pdf_to_txt_img_metadata,
            ocr_time,
            text_embed_time,
            img_embed_time,
        )
