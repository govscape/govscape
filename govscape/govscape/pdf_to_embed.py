import os
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
import fitz
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizer
from sentence_transformers import LoggingHandler
import torch
from pathlib import Path
import io
import numpy as np
from abc import ABC, abstractmethod
import json
from multiprocessing import get_context
import time
import re
import math
import logging
import shutil
import multiprocessing as mp
import pypdfium2
from .pdf_to_embed_multigpu import BGE_TextEmbeddingModel, ST_TextEmbeddingModel, compute_text_embeddings
from .indexing import WhooshIndex

# global vars *******************************************************************************************************
GPU_BATCH_SIZE = 2
BATCH_SIZE = 64
MAX_PDF_LENGTH = 50
# *******************************************************************************************************************

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

class EmbeddingModel(ABC):
    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_image(self, jpg_path):
        pass

class CLIPEmbeddingModel(EmbeddingModel):
    def __init__(self):
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)  # online
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")  # online
        self.processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)  # online
        self.model.eval()
        self.d = 512

    def encode_text(self, text):  #note: not doing encode_texts version of this yet because currently not in use. 
        #tokenize text        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)  # online
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        tokenized_text = inputs['input_ids'][0]

        #CLIP token limit = 77 so we have to divide into chunks and get embeddings for each of those
        max_chunk_len = 77
        text_chunks = []
        for i in range(0,len(tokenized_text), max_chunk_len):
            if len(tokenized_text[i:i+max_chunk_len]) == max_chunk_len or len(text_chunks) == 0:
                text_chunks.append(tokenized_text[i:i+max_chunk_len])

        #stack them all into a single batch so we can compute them all at the same time
        chunk_tensors = [] 
        for chunk in text_chunks:
            chunk_tensors.append(chunk.unsqueeze(0))
        batch_input_ids = torch.cat(chunk_tensors, dim=0)  
        batch_attention_mask = torch.ones_like(batch_input_ids)

        with torch.no_grad():
            batch_embeddings = self.model.get_text_features(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        embeddings = batch_embeddings.split(1, dim=0) 

        #decision: average embedding to create one embedding per PDF 
        final_embedding = torch.mean(torch.stack(embeddings), dim=0).to("cpu").numpy()

        return final_embedding

    def encode_image(self, jpg_path):
        image = Image.open(jpg_path).convert("RGB")

        # preprocess image
        inputs = self.processor(images = image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embedding = self.model.get_image_features(**inputs)
        
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        return image_embedding[0].to("cpu").numpy()
    
    # single gpu case 
    # def encode_images(self, jpg_paths):
        # batch_size = 32
        # images = []
        # for jpg_path in jpg_paths:
        #     img = Image.open(jpg_path)
        #     images.append(img)
        
        # inputs = self.processor(images=images, return_tensors="pt")
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
        # with torch.no_grad():
        #     if isinstance(self.model, torch.nn.DataParallel):  #multi-gpu case
        #         embeddings = self.model.module.get_image_features(**inputs)
        #     else:
        #         embeddings = self.model.get_image_features(**inputs)
        #     embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # return embeddings.cpu().numpy()
    
    # for multi-gpus: 
    def encode_images_per_gpu(self, input_batches, gpu_id, results):
        device = torch.device(f"cuda:{gpu_id}")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # online
        model.eval()
        all_embeddings = []

        for batch in input_batches:
            try:
                batch = {k : v.to(device) for k, v in batch.items()}  # move to gpu
                with torch.no_grad():
                    embeddings = model.get_image_features(**batch)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu())

            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

        if len(all_embeddings) > 0:
            results[gpu_id] = torch.cat(all_embeddings, dim=0).numpy()
        else:
            results[gpu_id] = np.empty((0, self.d), dtype=np.float32)

    @staticmethod
    def preprocess_image_batch(jpg_paths, processor):
        images = []
        for jpg_path in jpg_paths:
            try:
                img = Image.open(jpg_path)
                images.append(img)
            except Exception as e:
                print(e)
        input_batch = processor(images=images, return_tensors="pt", input_data_format="channels_last")
        return input_batch

    def encode_images(self, jpg_paths, max_batch_size=1024):
        gpu_count = torch.cuda.device_count()

        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)  # online
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")  # online
        processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)

        jpg_paths_batches = np.array_split(jpg_paths, math.ceil(len(jpg_paths) / max_batch_size))
        inputs = []
        ctx = get_context('spawn')
        with ctx.Pool(processes=os.cpu_count()) as pool:
            input_batches = pool.starmap(self.preprocess_image_batch, zip(jpg_paths_batches, [processor] * len(jpg_paths_batches)))
            inputs.extend(input_batches)
        
        manager = mp.Manager()
        outputs = manager.dict()
        processes = []
        ctx = get_context('spawn')
        print("starting processes for each gpu")
        for i in range(gpu_count):
            p = ctx.Process(target=self.encode_images_per_gpu, args=(inputs[math.ceil(i*len(inputs)/gpu_count):math.ceil((i+1)*len(inputs)/gpu_count)], i, outputs))
            p.start()
            processes.append(p)
        
        print("started processes for each gpu")
        for p in processes:
            p.join()
        
        all_embeddings = []
        for i in range(gpu_count):
            embeddings = outputs[i]
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

class PDFsToEmbeddings:
    def __init__(self, pdf_directory, data_dir, text_model, model_pool):
        self.pdfs_path = pdf_directory
        self.txts_path = data_dir + "/txt"
        self.img_path = data_dir + "/img"
        self.extracted_img_path = data_dir + "/img_extracted"
        self.embeddings_path = data_dir + "/embeddings"
        self.embeddings_img_path = data_dir + "/embeddings_img_pg"
        self.embeddings_img_e_path = data_dir + "/embeddings_img_extracted"
        self.index_keyword_directory = data_dir + "/index_keyword"
        self.metadata_dir = data_dir + "/metadata"
        self.text_model = text_model  # TextEmbeddingModel
        self.model_pool = model_pool  # for multi-gpu use, this is the model_pool

    # converts a single pdf file to txt and img files (one of each per page)
    @staticmethod
    def convert_pdf_to_txt_and_img(txts_path, imgs_path, pdfs_path, pdf_file):
        pdf_path = os.path.join(pdfs_path, pdf_file)
        pdf_txt_subdir = os.path.join(txts_path, os.path.splitext(pdf_file)[0])
        pdf_img_subdir = os.path.join(imgs_path, os.path.splitext(pdf_file)[0])

        os.makedirs(pdf_txt_subdir, exist_ok=True)
        os.makedirs(pdf_img_subdir, exist_ok=True)

        try:
            pdf = pypdfium2.PdfDocument(pdf_path)
            num_pages = len(pdf)
            if num_pages > MAX_PDF_LENGTH:
                return
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
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return
        
        for page_num, page_text in enumerate(text):
            txt_file_path = os.path.join(pdf_txt_subdir, f'{os.path.splitext(pdf_file)[0]}_{page_num}.txt')
            if page_text and len(page_text) != 0:
                with open(txt_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(page_text)
            
            img_file_path = os.path.join(pdf_img_subdir, f'{os.path.splitext(pdf_file)[0]}_{page_num}.jpeg')
            image = images[page_num]
            image.save(img_file_path, format="JPEG")


    # converts dir of pdfs -> dir of subdirs of txt files of each page AKA OVERALL PDFS -> TXTS
    def convert_pdfs_to_txt_and_img(self, pdf_files=None):
        self.ensure_dir(self.txts_path)
        if pdf_files is None:
            pdf_files = os.listdir(self.pdfs_path)
        ctx = get_context('forkserver')
        with ctx.Pool(processes=os.cpu_count()) as pool:
            pool.starmap(self.convert_pdf_to_txt_and_img, [(self.txts_path, self.img_path, self.pdfs_path, file) for file in pdf_files])

    # *******************************************************************************************************************
    # 1. this is the dir pdf -> dir img (of entire page) -> dir embed (of entire page) shared with og embed dir
    # *******************************************************************************************************************
    @staticmethod
    def convert_img_embedding_to_files_batch(embed_and_paths):
        embed, embed_file_paths = embed_and_paths
        for output_path, embedding in zip(embed_file_paths, embed):
            np.save(output_path, embedding)
    
    def convert_img_embedding_to_files(self, embed, embed_file_paths):
        # split the embedding up into chunks
        chunks = np.array_split(embed, os.cpu_count())
        # chunks = np.array_split(embed, 2)
        chunk_embed_file_paths = []
        start = 0
        for i in range(len(chunks)):
            end = chunks[i].shape[0]
            chunk_embed_file_paths.append(embed_file_paths[start: start + end])
            start = start + end 
        
        if len(chunks) != len(chunk_embed_file_paths):
            raise Exception("chunks and chunk_embed_file_paths should be the same length.")

        ctx = get_context('spawn')
        with ctx.Pool(processes=os.cpu_count()) as pool:
            pool.map(self.convert_img_embedding_to_files_batch, zip(chunks, chunk_embed_file_paths))
    
    # *******************************************************************************************************************
    # dir pdf --> dir img (extracted) -> dir embed (extracted) shared with og embed dir
    # *******************************************************************************************************************

    # single pdf -> extracted img, extracted img embedding (using og embed dir)  
    # multi gpu extract images below 
    @staticmethod
    def extract_img_pdfs(pdf_directory, extracted_img_path, embeddings_img_e_path, pdf_path):
        full_pdf_path = Path(pdf_directory) / Path(pdf_path)
        output_img_dir_path = Path(extracted_img_path) / Path(pdf_path).stem
        output_img_dir_path.mkdir(parents=True, exist_ok=True)

        try:
            with fitz.open(full_pdf_path) as pdf_doc:
                title = os.path.splitext(os.path.basename(pdf_path))[0]

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
                        except Exception as e:
                            continue

                        image_path = Path(output_img_dir_path) / f"{title}_{page_num}_{i}.jpeg"
                        image = image.convert("RGB")

                        if image.size[0] < 80 or image.size[1] < 80 or image.size[0] > 7000 or image.size[1] > 7000:  #image is too small/big to be considered
                            continue
                        image.save(image_path, "JPEG")
                
                if empty:
                    shutil.rmtree(output_img_dir_path)

        except Exception as e:
            logging.error(f"can't open PDF {pdf_path}: {e}")
            return

    
    def convert_pdfs_to_extracted_imgs(self, pdf_files):
        ctx = get_context('spawn')
        with ctx.Pool(processes=os.cpu_count()) as pool:
            pool.starmap(self.extract_img_pdfs, [(self.pdfs_path, self.extracted_img_path, self.embeddings_img_e_path, file) for file  in pdf_files])


    # *******************************************************************************************************************
    # pdf --> dir metadata (json) for each pdf
    # *******************************************************************************************************************
    def create_metadata_jsons(self, pdf_files):
        os.makedirs(self.metadata_dir, exist_ok=True)
        for pdf_file in pdf_files:
            json_data = dict()
            pdf_path = os.path.join(self.pdfs_path, pdf_file)
            try:
                pdf = pypdfium2.PdfDocument(pdf_path)
                num_pages = len(pdf)
                # pypdfium2 does not provide metadata directly, so set as Unknown or use another lib if needed
                gov_name = pdf.get_metadata_value("Title")
                timestamp = pdf.get_metadata_value("CreationDate")
                if len(gov_name) == 0:
                    gov_name = 'Unknown'
                if len(timestamp) == 0:
                    timestamp = 'Unknown'
                json_data['gov_name'] = gov_name
                json_data['timestamp'] = timestamp
                json_data['num_pages'] = num_pages
            except Exception as e:
                json_data['gov_name'] = 'Unknown'
                json_data['timestamp'] = 'Unknown'
                json_data['num_pages'] = 1
                print(f"Skipping invalid PDF {pdf_path}: {e}")
                continue
            
            pdf_metadata_dir = os.path.join(self.metadata_dir, os.path.splitext(pdf_file)[0])
            os.makedirs(pdf_metadata_dir, exist_ok=True)
            json_file_path = os.path.join(pdf_metadata_dir, "metadata.json")
            with open(json_file_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

    # *******************************************************************************************************************
    # keyword indexing
    # *******************************************************************************************************************

    def add_texts_to_whoosh_index(self, pdf_files):
        """Add text from the current batch of pdf_files to the WhooshIndex object."""
        whoosh_index = WhooshIndex(self.index_keyword_directory)
        texts = []
        pdf_names = []
        pages = []
        for pdf_file in pdf_files:
            pdf_txt_subdir = os.path.join(self.txts_path, os.path.splitext(pdf_file)[0])
            if not os.path.exists(pdf_txt_subdir):
                continue
            for txt_file in os.listdir(pdf_txt_subdir):
                txt_path = os.path.join(pdf_txt_subdir, txt_file)
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    if text.strip():
                        texts.append(text)
                        pdf_names.append(os.path.splitext(pdf_file)[0])
                        # Extract page number from filename (assumes format: <pdfname>_<page>.txt)
                        page_num = int(os.path.splitext(txt_file)[0].split('_')[-1])
                        pages.append(page_num)
                except Exception as e:
                    print(f"Error reading {txt_path}: {e}")
        if texts:
            whoosh_index.add_batch(texts, pdf_names, pages)
            print(f"Added {len(texts)} pages to Whoosh keyword index.")
        else:
            print("No text pages found to add to Whoosh keyword index.")

    # *******************************************************************************************************************
    # overall pipeline
    # *******************************************************************************************************************

    def pdfs_to_embeddings(self, pdf_files=None):
        pdf_files = pdf_files or os.listdir(self.pdfs_path)
        time1 = time.time()

        print("Converting pdfs to txts and page images")
        self.convert_pdfs_to_txt_and_img(pdf_files)
        time2 = time.time()

        print("Converting txts to embeddings")
        compute_text_embeddings(self.text_model, self.model_pool, self.txts_path, self.embeddings_path)
        time3 = time.time()
        
        os.makedirs(self.embeddings_img_path, exist_ok=True)
        img_paths = []
        embedding_paths = []
        for img_subdir in os.scandir(self.img_path):
            if img_subdir.is_dir():
                os.makedirs(os.path.join(self.embeddings_img_path, img_subdir.name), exist_ok=True)
                img_subdir_paths = os.listdir(img_subdir.path)
                for img_file in img_subdir_paths:
                    img_paths.append(os.path.join(img_subdir.path, img_file))
                    embedding_paths.append(os.path.join(self.embeddings_img_path,  img_subdir.name, os.path.splitext(img_file)[0] + '.npy'))

        print("Embedding this many images: ", len(img_paths))
        img_model = CLIPEmbeddingModel()
        emb = img_model.encode_images(img_paths)

        print("Embeddings computed. Shape:", emb.shape)
        self.convert_img_embedding_to_files(emb, embedding_paths)
        time4 = time.time()

        # TODO: Remove metadata jsons in favor of SQLite Database
        print("Creating metadata jsons for each pdf")
        self.create_metadata_jsons(pdf_files)  # extract images and save
        time5 = time.time()

        pdf_to_txt_img = time2 - time1
        text_embed_time = time3 - time2
        img_embed_time = time4 - time3
        metadata_time = time5 - time4

        print("pdf -> txt and img time: ", pdf_to_txt_img)
        print("txt -> embed time: ", text_embed_time)
        print("img per page -> embed time: ", img_embed_time)
        print("pdf -> json time: ", metadata_time)

        return pdf_to_txt_img, text_embed_time, img_embed_time, metadata_time

    # *******************************************************************************************************************
    # helper functions
    # *******************************************************************************************************************

    # makes sure that the directory specified is created
    @staticmethod
    def ensure_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
