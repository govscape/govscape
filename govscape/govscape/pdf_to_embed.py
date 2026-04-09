import io
import json
import logging
import os
import re
import shutil
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import fitz
import pypdfium2
from PIL import Image, ImageFile
from sentence_transformers import LoggingHandler

# New imports for OCR and OCR Engines
from .ocr_engines import OlmOCREngine, OCRmyPDFEngine, PaddleOCREngine, EasyOCREngine, OCREngine

from .text_embedding_models import (
    BGE_TextEmbeddingModel,
    BGESmall_TextEmbeddingModel,
    Dummy_TextEmbeddingModel,
    ST_TextEmbeddingModel,
)
from .utils import read_txt_file
from .visual_embedding_models import (
    CLIP_VisualEmbeddingModel,
    Dummy_VisualEmbeddingModel,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# global vars ---------------------------------------------------------------
GPU_BATCH_SIZE = 2
BATCH_SIZE = 64
# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def natural_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


class PDFsToEmbeddings:
    def __init__(self, pdf_directory, data_dir, text_model_type, visual_model_type, ocr_engine_type=" OlmOCR"):
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

	if ocr_engine_type == "OlmOCR":
	    self.ocr_engine = OlmOCREngine()
	elif ocr_engine_type == "OCRmyPDF":
	    self.ocr_engine = OCRmyPDFEngine()
	elif ocr_engine_type == "PaddleOCR":
	    self.ocr_engine = PaddleOCREngine()
	elif ocr_engine_type == "EasyOCR":
	    self.ocr_engine = EasyOCREngine()
	elif ocr_engine_type == "None":
	    self.ocr_engine = None
	else:
	    raise ValueError("Unsupported OCR engine type")

    # converts a single pdf file to txt and img files (one of each per page)
    @staticmethod
    def convert_pdf_to_txt_img_and_metadata(
        txts_path, imgs_path, pdfs_path, metadata_dir, pdf_file
    ):
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

            # Create Metadata JSON
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

            # Gather text and image for each page
            text = []
            images = []
            for i in range(num_pages):
                page = pdf[i]
                # Extract text
                page_text = page.get_textpage().get_text_bounded()
                text.append(page_text)
                # Render image
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

    # converts dir of pdfs -> dir of subdirs of txt files of each page
    # AKA OVERALL PDFS -> TXTS
    def convert_pdfs_to_txt_img_and_metadata(self, pdf_files):
        if not os.path.exists(self.txts_path):
            os.makedirs(self.txts_path)
        ctx = get_context("forkserver")
        with ctx.Pool(processes=self.cpu_count) as pool:
            results = pool.starmap(
                self.convert_pdf_to_txt_img_and_metadata,
                [
                    (
                        self.txts_path,
                        self.img_path,
                        self.pdfs_path,
                        self.metadata_dir,
                        file,
                    )
                    for file in pdf_files
                ],
            )
        return sum(results)

    # Create a specific directory for this PDF's OCR output
    # Perform OCR and get text content
    # Save the OCR results as a single text file for the entire document
    # The structure for extracted images/text per page is different.
    def _perform_ocr_on_file(ocr_engine: OCREngine, pdfs_path: str, txts_path: str, pdf_file: str) -> bool:
	pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
        pdf_path = os.path.join(pdfs_path, pdf_file)
        # Create a specific directory for this PDF's OCR output
        pdf_ocr_txt_subdir = os.path.join(txts_path, pdf_name)
        os.makedirs(pdf_ocr_txt_subdir, exist_ok=True)

	try:
	    logging.info(f"Starting OCR on {pdf_file} using {ocr_engine.name}")
	    ocr_text = ocr_engine.process(pdf_path)
            ocr_file_path = os.path.join(pdf_ocr_txt_subdir, f"{pdf_name}_ocr.txt")
	    with open(ocr_file_path, "w", encoding="utf-8") as text_file:
	        text_file.write(ocr_text)
	    logging.info(f"Completed OCR on {pdf_file}")
	    return True
	except Exception as e:
	    logging.error(f"Failed to OCR {pdf_file}: {e}")
	    return False


    def convert_pdfs_to_ocr_txts(self, pdf_files):
        """
        Uses multiprocessing to perform OCR on all given PDF files.
        """
        if not self.ocr_engine:
            logging.warning("OCR Engine not initialized, skipping OCR stage.")
            return 0
            
        if not os.path.exists(self.txts_path):
            os.makedirs(self.txts_path)
            
        ctx = get_context("spawn") # Use "spawn" for safer multiprocessing with heavy libraries
        with ctx.Pool(processes=self.cpu_count) as pool:
            # Prepare arguments: (self.ocr_engine, self.pdfs_path, self.txts_path, file) for each pdf
            results = pool.starmap(
                self._perform_ocr_on_file,
                [
                    (
                        self.ocr_engine,
                        self.pdfs_path,
                        self.txts_path,
                        file,
                    )
                    for file in pdf_files
                ],
            )
        return sum(results)

    # ----------------------------------------
    # 1. dir pdf -> dir img (entire page) -> dir embed (entire page) shared with og
    #    embed dir
    # ----------------------------------------
    @staticmethod
    def _convert_img_embedding_to_files_batch(embed_and_paths):
        embed, embed_file_paths = embed_and_paths
        for output_path, embedding in zip(embed_file_paths, embed, strict=False):
            np.save(output_path, embedding)

    def convert_img_embedding_to_files(self, embed, embed_file_paths):
        # split the embedding up into chunks
        chunks = np.array_split(embed, self.cpu_count)
        # chunks = np.array_split(embed, 2)
        chunk_embed_file_paths = []
        start = 0
        for i in range(len(chunks)):
            end = chunks[i].shape[0]
            chunk_embed_file_paths.append(embed_file_paths[start : start + end])
            start = start + end

        if len(chunks) != len(chunk_embed_file_paths):
            raise Exception(
                "chunks and chunk_embed_file_paths should be the same length."
            )

        ctx = get_context("spawn")
        with ctx.Pool(processes=self.cpu_count) as pool:
            pool.map(
                self._convert_img_embedding_to_files_batch,
                zip(chunks, chunk_embed_file_paths, strict=False),
            )

    # ----------------------------------------
    # dir pdf -> dir img (extracted) -> dir embed (extracted) shared with og
    # embed dir
    # ----------------------------------------

    # single pdf -> extracted img, extracted img embedding (using og embed dir)
    # multi gpu extract images below
    @staticmethod
    def extract_img_pdfs(
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
                        if i == 4:  # four images max per page extracted
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
                            Path(output_img_dir_path)
                            / f"{pdf_name}_{page_num}_{i}.jpeg"
                        )
                        image = image.convert("RGB")

                        if (
                            image.size[0] < 80
                            or image.size[1] < 80
                            or image.size[0] > 7000
                            or image.size[1] > 7000
                        ):  # image is too small/big to be considered
                            continue
                        image.save(image_path, "JPEG")

                if empty:
                    shutil.rmtree(output_img_dir_path)

        except Exception as e:
            logging.error(f"can't open PDF {pdf_path}: {e}")
            return

    def convert_pdfs_to_extracted_imgs(self, pdf_files):
        ctx = get_context("spawn")
        with ctx.Pool(processes=self.cpu_count) as pool:
            pool.starmap(
                self.extract_img_pdfs,
                [
                    (
                        self.pdfs_path,
                        self.extracted_img_path,
                        self.embeddings_img_e_path,
                        file,
                    )
                    for file in pdf_files
                ],
            )

    # ----------------------------------------
    # Create Text Embeddings
    # ----------------------------------------
    @staticmethod
    def create_metadata_jsons_worker(pdf_files, metadata_dir):
        for pdf_file in pdf_files:
            json_data = dict()
            pdf_path = os.path.join(pdf_file)
            try:
                pdf = pypdfium2.PdfDocument(pdf_path)
                num_pages = len(pdf)
                # pypdfium2 does not provide metadata directly, so set as Unknown or use another lib if needed
                gov_name = pdf.get_metadata_value("Title")
                timestamp = pdf.get_metadata_value("CreationDate")
                keywords = pdf.get_metadata_value("Keywords")
                if len(keywords) == 0:
                    keywords = 'Unknown'
                if len(gov_name) == 0:
                    gov_name = 'Unknown'
                if len(timestamp) == 0:
                    timestamp = 'Unknown'
                json_data['gov_name'] = gov_name
                json_data['timestamp'] = timestamp
                json_data['num_pages'] = num_pages
                json_data['keywords'] = keywords
            except Exception as e:
                json_data['gov_name'] = 'Unknown'
                json_data['timestamp'] = 'Unknown'
                json_data['num_pages'] = 1
                json_data['keywords'] = 'Unknown'
                print(f"Skipping invalid PDF {pdf_path}: {e}")
                continue

            pdf_metadata_dir = os.path.join(metadata_dir, os.path.splitext(os.path.basename(pdf_file))[0])
            os.makedirs(pdf_metadata_dir, exist_ok=True)
            json_file_path = os.path.join(pdf_metadata_dir, "metadata.json")
            with open(json_file_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

    def convert_subdirs_to_embeddings(txt_path, embed_path):
        if not os.path.exists(embed_path):
            os.makedirs(embed_path)

        txt_subdirs_paths = [
            txt_subdir.path
            for txt_subdir in os.scandir(txt_path)
            if txt_subdir.is_dir()
        ]

        text_batch = []
        file_batch = []
        for txt_subdir_path in txt_subdirs_paths:
            embed_name = os.path.basename(txt_subdir_path)
            embedding_dir = os.path.join(embed_path, embed_name)

            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)

            # Check if an OCR output file exists for this PDF
            ocr_file_path = os.path.join(txt_subdir_path, f"{embed_name}_ocr.txt")
            if os.path.exists(ocr_file_path):
                # Use the OCR text for embeddings
                logging.info(f"Using OCR output for embedding {embed_name}")
                text = read_txt_file(ocr_file_path)
                output_path = os.path.join(embedding_dir, f"{embed_name}_ocr.npy")
                text_batch.append(text)
                file_batch.append(output_path)
            else:
                # Use the original per-page text files
                logging.info(f"Using standard text extraction for embedding {embed_name}")
                txt_files = [f for f in os.listdir(txt_subdir_path) if f.endswith(".txt")]
                
                for txt_file in txt_files:
                    txt_path = os.path.join(txt_subdir_path, txt_file)
                    text = read_txt_file(txt_path)
                    output_path = os.path.join(
                        embedding_dir, txt_file.replace(".txt", ".npy")
                    )
                    text_batch.append(text)
                    file_batch.append(output_path)

        return text_batch, file_batch

    @staticmethod
    def convert_embedding_to_files(embeddings, embed_file_paths):
        for embedding, output_path in zip(embeddings, embed_file_paths, strict=False):
            np.save(output_path, embedding)

    def compute_text_embeddings(self):
        # sentences
        sentences, all_embed_file_paths = self.convert_subdirs_to_embeddings(
            self.txts_path, self.embeddings_path
        )

        embeddings = self.text_model.encode_text_batch(sentences)
        print("Embeddings computed. Shape:", embeddings.shape)

        # put them into embedding files
        self.convert_embedding_to_files(embeddings, all_embed_file_paths)

    # -------------------------------------------------------------------------------
    # Create Page Image Embeddings
    # -------------------------------------------------------------------------------
    def compute_image_embeddings(self):
        os.makedirs(self.embeddings_img_path, exist_ok=True)
        img_paths = []
        embedding_paths = []
        for img_subdir in os.scandir(self.img_path):
            if img_subdir.is_dir():
                os.makedirs(
                    os.path.join(self.embeddings_img_path, img_subdir.name),
                    exist_ok=True,
                )
                img_subdir_paths = os.listdir(img_subdir.path)
                for img_file in img_subdir_paths:
                    img_paths.append(os.path.join(img_subdir.path, img_file))
                    embedding_paths.append(
                        os.path.join(
                            self.embeddings_img_path,
                            img_subdir.name,
                            os.path.splitext(img_file)[0] + ".npy",
                        )
                    )

        print("Embedding this many images: ", len(img_paths))
        emb = self.visual_model.encode_images(img_paths)

        print("Embeddings computed. Shape:", emb.shape)
        self.convert_img_embedding_to_files(emb, embedding_paths)

    # -------------------------------------------------------------------------------
    # overall pipeline
    # -------------------------------------------------------------------------------
    def pdfs_to_embeddings(
        self, pdf_files, do_text_embedding, do_img_embedding, do_metadata_collection, do_ocr=False
    ):
        """
        The main method for the pipeline, now including an OCR stage.
        """
        time1 = time.time()
        pdfs_successfully_parsed = 0
        if do_text_embedding or do_img_embedding or do_metadata_collection:
            print("Converting pdfs to txts and page images")
            pdfs_successfully_parsed = self.convert_pdfs_to_txt_img_and_metadata(
                pdf_files
            )
        print(
            f"% pdfs successfully parsed: {pdfs_successfully_parsed} / {len(pdf_files)}"
        )
        
        time2 = time.time()

        # --- New Stage 2: OCR ---
        pdfs_successfully_ocred = 0
        if do_ocr:
            print(f"Applying {self.ocr_engine.name if self.ocr_engine else 'Unknown'} OCR to PDFs")
            pdfs_successfully_ocred = self.convert_pdfs_to_ocr_txts(pdf_files)
            print(f"% pdfs successfully OCRed: {pdfs_successfully_ocred} / {len(pdf_files)}")
        # ------------------------

        time3 = time.time()

        if do_text_embedding:
            print("Converting txts to embeddings")
            # compute_text_embeddings internally uses convert_subdirs_to_embeddings, 
            # which we have modified to look for OCR files.
            self.compute_text_embeddings()
        
        time4 = time.time()

        if do_img_embedding:
            self.compute_image_embeddings()

        time5 = time.time()

        # Calculate durations for reporting
        pdf_to_txt_img_metadata_time = time2 - time1
        ocr_time = time3 - time2
        text_embed_time = time4 - time3
        img_embed_time = time5 - time4

        print("pdf -> txt, img, metadata time: ", pdf_to_txt_img_metadata_time)
        print("ocr time: ", ocr_time)
        print("txt -> embed time: ", text_embed_time)
        print("img per page -> embed time: ", img_embed_time)

        return (
            pdf_to_txt_img_metadata_time,
            ocr_time,
            text_embed_time,
            img_embed_time,
        )
