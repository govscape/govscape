import functools
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

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

    def extract(self, pdf_files: list[str], data_model: DataModel) -> float:
        """Extract page text, images and metadata from pdf_files into data_model.
        With an ocr_type, also render the pages that need OCR, for ocr(). Returns
        seconds taken."""
        start = time.time()
        logging.info("Converting pdfs to txts and page images")
        pdf_extraction_stage = PDFExtractionStage(
            data_model=data_model,
            pdf_files=pdf_files,
            cpu_count=self.cpu_count,
            render_ocr_pages=bool(self.ocr_type),
        )
        pdf_extraction_stage.validate()
        parsed = pdf_extraction_stage.run()
        logging.info(f"PDFs successfully parsed: {parsed} / {len(pdf_files)}")
        elapsed = time.time() - start
        logging.info(f"pdf -> txt, img, metadata time: {elapsed}")
        return elapsed

    def ocr(self, data_model: DataModel) -> float:
        """OCR the pages extract() rendered for it, if this pipeline has an
        ocr_type. Returns seconds taken."""
        if not self.ocr_type:
            return 0.0
        start = time.time()
        logging.info(f"Running OCR extraction using {self.ocr_type}")
        ocr_stage = OCRProcessingStage(
            data_model=data_model, ocr_type=self.ocr_type, **self.ocr_kwargs
        )
        ocr_stage.validate()
        ocr_stage.run()
        elapsed = time.time() - start
        logging.info(f"ocr processing time: {elapsed}")
        return elapsed

    def embed(
        self, data_model: DataModel, do_text_embedding: bool, do_img_embedding: bool
    ) -> tuple[float, float]:
        """Embed the extracted pages in data_model. Returns (text, image) seconds."""
        if do_text_embedding and do_img_embedding and self._text_model.SERVED_REMOTELY:
            # The image model can use the GPU while the text model's server works.
            with ThreadPoolExecutor(max_workers=1) as executor:
                text_future = executor.submit(self._embed_text, data_model)
                image_time = self._embed_images(data_model)
            return text_future.result(), image_time
        text_time = self._embed_text(data_model) if do_text_embedding else 0.0
        image_time = self._embed_images(data_model) if do_img_embedding else 0.0
        return text_time, image_time

    def _embed_text(self, data_model: DataModel) -> float:
        start = time.time()
        logging.info("Converting txts to embeddings")
        text_stage = TextEmbeddingStage(data_model, model=self._text_model)
        text_stage.validate()
        text_stage.run()
        elapsed = time.time() - start
        logging.info(f"txt -> embed time: {elapsed}")
        return elapsed

    def _embed_images(self, data_model: DataModel) -> float:
        start = time.time()
        logging.info("Converting imgs to embeddings")
        image_stage = PageImageEmbeddingStage(
            data_model, cpu_count=self.cpu_count, model=self._visual_model
        )
        image_stage.validate()
        image_stage.run()
        elapsed = time.time() - start
        logging.info(f"img per page -> embed time: {elapsed}")
        return elapsed

    def process_pdfs(
        self,
        pdf_files: list[str],
        do_text_embedding: bool,
        do_img_embedding: bool,
        do_metadata_collection: bool,
    ):
        """Extract, OCR (with an ocr_type) and embed pdf_files into this pipeline's
        data directory."""
        extract_time = ocr_time = 0.0
        if (
            do_text_embedding
            or do_img_embedding
            or do_metadata_collection
            or self.ocr_type
        ):
            extract_time = self.extract(pdf_files, self.data_model)
            ocr_time = self.ocr(self.data_model)
        text_embed_time, img_embed_time = self.embed(
            self.data_model, do_text_embedding, do_img_embedding
        )
        return extract_time, ocr_time, text_embed_time, img_embed_time
