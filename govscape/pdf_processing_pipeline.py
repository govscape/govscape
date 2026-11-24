import functools
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
from .processing.page_image_embedding_stage import build_visual_model
from .processing.text_embedding_stage import build_text_model


class PDFProcessingPipeline:
    def __init__(
        self,
        data_dir: str,
        text_model_type: str,
        visual_model_type: str,
        ocr_type: str | None = None,
        **ocr_kwargs,
    ):
        self.data_model = DataModel(data_dir)
        self.cpu_count = os.cpu_count() or 1
        self.text_model_type = text_model_type
        self.visual_model_type = visual_model_type
        self.ocr_type = ocr_type
        self.ocr_kwargs = ocr_kwargs

    # The models are loaded once and reused for every batch.
    @functools.cached_property
    def _text_model(self):
        return build_text_model(self.text_model_type)

    @functools.cached_property
    def _visual_model(self):
        return build_visual_model(self.visual_model_type)

    def extract(
        self, pdf_files: list[str], data_model: DataModel, do_ocr: bool = False
    ) -> tuple[float, float]:
        """Extract page text, images and metadata from pdf_files into data_model,
        then OCR the pages that need it. Returns (extraction, OCR) seconds."""
        time1 = time.time()
        logging.info("Converting pdfs to txts and page images")
        pdf_extraction_stage = PDFExtractionStage(
            data_model=data_model, pdf_files=pdf_files, cpu_count=self.cpu_count
        )
        pdf_extraction_stage.validate()
        parsed = pdf_extraction_stage.run()
        logging.info(f"PDFs successfully parsed: {parsed} / {len(pdf_files)}")

        time2 = time.time()
        if do_ocr and not self.ocr_type:
            logging.warning("do_ocr=True but no ocr_type set; skipping OCR stage")
        if do_ocr and self.ocr_type:
            logging.info(f"Running OCR extraction using {self.ocr_type}")
            ocr_stage = OCRProcessingStage(
                data_model=data_model, ocr_type=self.ocr_type, **self.ocr_kwargs
            )
            ocr_stage.validate()
            ocr_stage.run()
        time3 = time.time()

        logging.info(f"pdf -> txt, img, metadata time: {time2 - time1}")
        if do_ocr:
            logging.info(f"ocr processing time: {time3 - time2}")
        return time2 - time1, time3 - time2

    def embed(
        self, data_model: DataModel, do_text_embedding: bool, do_img_embedding: bool
    ) -> tuple[float, float]:
        """Embed the extracted pages in data_model. Returns (text, image) seconds."""
        time1 = time.time()
        if do_text_embedding:
            logging.info("Converting txts to embeddings")
            text_stage = TextEmbeddingStage(data_model, model=self._text_model)
            text_stage.validate()
            text_stage.run()
        time2 = time.time()

        if do_img_embedding:
            logging.info("Converting imgs to embeddings")
            image_stage = PageImageEmbeddingStage(
                data_model, cpu_count=self.cpu_count, model=self._visual_model
            )
            image_stage.validate()
            image_stage.run()
        time3 = time.time()

        logging.info(f"txt -> embed time: {time2 - time1}")
        logging.info(f"img per page -> embed time: {time3 - time2}")
        return time2 - time1, time3 - time2

    def process_pdfs(
        self,
        pdf_files: list[str],
        do_text_embedding: bool,
        do_img_embedding: bool,
        do_metadata_collection: bool,
        do_ocr: bool = False,
    ):
        """Extract and embed pdf_files into this pipeline's data directory."""
        extract_time = ocr_time = 0.0
        if do_text_embedding or do_img_embedding or do_metadata_collection or do_ocr:
            extract_time, ocr_time = self.extract(pdf_files, self.data_model, do_ocr)
        text_embed_time, img_embed_time = self.embed(
            self.data_model, do_text_embedding, do_img_embedding
        )
        return extract_time, ocr_time, text_embed_time, img_embed_time
