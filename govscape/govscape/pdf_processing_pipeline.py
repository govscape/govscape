import logging
import os
import time

from sentence_transformers import LoggingHandler

from .processing import (
    PageImageEmbeddingStage,
    PDFExtractionStage,
    TextEmbeddingStage,
)
from .text_embedding_models import (
    BGE_TextEmbeddingModel,
    BGESmall_TextEmbeddingModel,
    Dummy_TextEmbeddingModel,
    ST_TextEmbeddingModel,
)
from .visual_embedding_models import (
    CLIP_VisualEmbeddingModel,
    Dummy_VisualEmbeddingModel,
)

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class PDFProcessingPipeline:
    def __init__(self, pdf_directory, data_dir, text_model_type, visual_model_type):
        self.pdfs_path = pdf_directory
        self.txts_path = data_dir + "/txt"
        self.img_path = data_dir + "/img"
        self.extracted_img_path = data_dir + "/img_extracted"
        self.embeddings_path = data_dir + "/embeddings"
        self.embeddings_img_path = data_dir + "/embeddings_img_pg"
        self.embeddings_img_e_path = data_dir + "/embeddings_img_extracted"
        self.index_keyword_directory = data_dir + "/index_keyword"
        self.metadata_dir = data_dir + "/metadata"
        self.cpu_count = os.cpu_count()

        if text_model_type == "ST":
            self.text_model = ST_TextEmbeddingModel()
        elif text_model_type == "BGE":
            self.text_model = BGE_TextEmbeddingModel()
        elif text_model_type == "BGESmall":
            self.text_model = BGESmall_TextEmbeddingModel()
        elif text_model_type == "Dummy":
            self.text_model = Dummy_TextEmbeddingModel()
        else:
            raise ValueError("Unsupported model type")

        if visual_model_type == "CLIP":
            self.visual_model = CLIP_VisualEmbeddingModel()
        elif visual_model_type == "Dummy":
            self.visual_model = Dummy_VisualEmbeddingModel()
        else:
            raise ValueError("Unsupported model type")

    def pdfs_to_embeddings(
        self, pdf_files, do_text_embedding, do_img_embedding, do_metadata_collection
    ):
        pdf_extraction_stage = PDFExtractionStage(
            pdfs_path=self.pdfs_path,
            txts_path=self.txts_path,
            img_path=self.img_path,
            metadata_dir=self.metadata_dir,
            pdf_files=pdf_files,
            cpu_count=self.cpu_count,
        )
        text_embedding_stage = TextEmbeddingStage(
            txts_path=self.txts_path,
            embeddings_path=self.embeddings_path,
            text_model=self.text_model,
        )
        page_image_embedding_stage = PageImageEmbeddingStage(
            img_path=self.img_path,
            embeddings_img_path=self.embeddings_img_path,
            visual_model=self.visual_model,
            cpu_count=self.cpu_count,
        )

        run_extraction = do_text_embedding or do_img_embedding or do_metadata_collection
        errors = []
        if run_extraction:
            errors.extend(pdf_extraction_stage.validate())
        if do_text_embedding and not run_extraction:
            errors.extend(text_embedding_stage.validate())
        if do_img_embedding and not run_extraction:
            errors.extend(page_image_embedding_stage.validate())
        if errors:
            raise ValueError("Pipeline validation failed:\n" + "\n".join(errors))

        time1 = time.time()
        pdfs_successfully_parsed = 0
        if do_text_embedding or do_img_embedding or do_metadata_collection:
            print("Converting pdfs to txts and page images")
            pdfs_successfully_parsed = pdf_extraction_stage.run()
        print(
            f"% pdfs successfully parsed: {pdfs_successfully_parsed} / {len(pdf_files)}"
        )

        time2 = time.time()
        if do_text_embedding:
            print("Converting txts to embeddings")
            text_embedding_stage.run()
        time3 = time.time()

        if do_img_embedding:
            page_image_embedding_stage.run()
        time4 = time.time()

        pdf_to_txt_img_metadata = time2 - time1
        text_embed_time = time3 - time2
        img_embed_time = time4 - time3

        print("pdf -> txt, img, metadata time: ", pdf_to_txt_img_metadata)
        print("txt -> embed time: ", text_embed_time)
        print("img per page -> embed time: ", img_embed_time)

        return (
            pdf_to_txt_img_metadata,
            text_embed_time,
            img_embed_time,
        )
