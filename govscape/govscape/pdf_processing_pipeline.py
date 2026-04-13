import logging
import os
import time

from .processing import (
    PageImageEmbeddingStage,
    PDFExtractionStage,
    TextEmbeddingStage,
)


class PDFProcessingPipeline:
    def __init__(
        self,
        pdf_directory: str,
        data_dir: str,
        text_model_type: str,
        visual_model_type: str,
    ):
        self.pdfs_path = pdf_directory
        self.txts_path = data_dir + "/txt"
        self.img_path = data_dir + "/img"
        self.embeddings_path = data_dir + "/embeddings"
        self.embeddings_img_path = data_dir + "/embeddings_img_pg"
        self.index_keyword_directory = data_dir + "/index_keyword"
        self.metadata_dir = data_dir + "/metadata"
        self.cpu_count = os.cpu_count()
        self.text_model_type = text_model_type
        self.visual_model_type = visual_model_type

    def process_pdfs(
        self,
        pdf_files: list[str],
        do_text_embedding: bool,
        do_img_embedding: bool,
        do_metadata_collection: bool,
    ):

        run_extraction = do_text_embedding or do_img_embedding or do_metadata_collection

        time1 = time.time()
        pdfs_successfully_parsed = 0
        if run_extraction:
            logging.info("Converting pdfs to txts and page images")
            pdf_extraction_stage = PDFExtractionStage(
                pdfs_path=self.pdfs_path,
                txts_path=self.txts_path,
                img_path=self.img_path,
                metadata_dir=self.metadata_dir,
                pdf_files=pdf_files,
                cpu_count=self.cpu_count,
            )
            pdf_extraction_stage.validate()
            pdfs_successfully_parsed = pdf_extraction_stage.run()
        logging.info(
            f"PDFs successfully parsed: {pdfs_successfully_parsed} / {len(pdf_files)}"
        )

        time2 = time.time()
        if do_text_embedding:
            logging.info("Converting txts to embeddings")
            text_embedding_stage = TextEmbeddingStage(
                txts_path=self.txts_path,
                embeddings_path=self.embeddings_path,
                model_type=self.text_model_type,
            )
            text_embedding_stage.validate()
            text_embedding_stage.run()
        time3 = time.time()

        if do_img_embedding:
            logging.info("Converting imgs to embeddings")
            page_image_embedding_stage = PageImageEmbeddingStage(
                img_path=self.img_path,
                embeddings_img_path=self.embeddings_img_path,
                model_type=self.visual_model_type,
                cpu_count=self.cpu_count,
            )
            page_image_embedding_stage.validate()
            page_image_embedding_stage.run()
        time4 = time.time()

        pdf_to_txt_img_metadata = time2 - time1
        text_embed_time = time3 - time2
        img_embed_time = time4 - time3

        logging.info(f"pdf -> txt, img, metadata time: {pdf_to_txt_img_metadata}")
        logging.info(f"txt -> embed time: {text_embed_time}")
        logging.info(f"img per page -> embed time: {img_embed_time}")

        return (
            pdf_to_txt_img_metadata,
            text_embed_time,
            img_embed_time,
        )
