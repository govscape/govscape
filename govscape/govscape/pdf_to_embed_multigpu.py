import os
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sentence_transformers import SentenceTransformer, LoggingHandler
import torch
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import get_context
import math
import re
import logging
import pynvml

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

# global vars
GPU_BATCH_SIZE = 16
BATCH_SIZE = 128

class EmbeddingModel(ABC):
    @abstractmethod
    def encode_text(self, text):
        pass

    @abstractmethod
    def encode_image(self, jpg_path):
        pass

def get_least_used_cuda():
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        min_used_mem = float("inf")
        best_device = "cuda:0"
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if meminfo.used < min_used_mem:
                min_used_mem = meminfo.used
                best_device = f"cuda:{i}"
        pynvml.nvmlShutdown()
        return best_device

class ST_TextEmbeddingModel(EmbeddingModel):
    def __init__(self):
        self.device = get_least_used_cuda() if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') 
        self.d = self.model.get_sentence_embedding_dimension()
    
    def encode_text(self, text, is_query=False):
        if self.model.device != self.device:
            self.model.to(self.device)
        with torch.no_grad():
            embedding = self.model.encode(text, batch_size=GPU_BATCH_SIZE, device=self.device)
        return embedding
    
    def encode_image(self, jpg_path):
        raise NotImplementedError("TextEmbeddingModel does not support image encoding.")
    
class BGE_TextEmbeddingModel(EmbeddingModel):
    def __init__(self):
        self.device = get_least_used_cuda() if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('BAAI/bge-base-en-v1.5') 
        self.d = self.model.get_sentence_embedding_dimension()
    
    def encode_text(self, text, is_query=False):
        if self.model.device != self.device:
            self.model.to(self.device)
        if is_query:
            text = "Represent this sentence for searching relevant passages:" + text
        with torch.no_grad():
            embedding = self.model.encode(text, batch_size=GPU_BATCH_SIZE, device=self.device)
        return embedding
    
    def encode_image(self, jpg_path):
        raise NotImplementedError("TextEmbeddingModel does not support image encoding.")
    
# for sorting file names with page numbers to ensure consistency when batching between txt and npy files (OS could 
    # order file names differently)
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

class TxtsToEmbeddings:
    def __init__(self, txt_directory, embeddings_dir):
        self.txts_path = txt_directory
        self.embeddings_path = embeddings_dir


    # (2) txt -> embed

    def txt_to_text(self, txt_path):
        text = ""
        with open(txt_path, 'r') as file:
            text = file.read()
        return text 
    
    # multiple txt subdir paths -> multiple embed dirs
    # set so the number of page files matches the batch size. 
    def convert_subdirs_to_embeddings(self, txt_subdir_paths):
        text_batch = []
        file_batch = []
        for txt_subdir_path in txt_subdir_paths:
            embed_name = os.path.basename(txt_subdir_path)
            embedding_dir = os.path.join(self.embeddings_path, embed_name)
            self.ensure_dir(embedding_dir)

            #all txt files in the txt subdir 
            txt_files = sorted(os.listdir(txt_subdir_path), key = natural_key)

            for txt_file in txt_files:
                txt_path = os.path.join(txt_subdir_path, txt_file)
                text = self.txt_to_text(txt_path)
                output_path = os.path.join(embedding_dir, txt_file)
                text_batch.append(text)
                file_batch.append(output_path)
        
        return text_batch, file_batch


    # 2. MP VERSION with pdf_files
    def convert_txts_to_embeddings(self):
        all_texts = []
        all_embed_file_paths = []
        self.ensure_dir(self.embeddings_path)

        txt_subdirs_paths = []
        for txt_subdir in os.scandir(self.txts_path):
            if txt_subdir.is_dir():
                txt_subdirs_paths.append(txt_subdir.path)
        
        # splitting into groups for each process:   # TODO: verify concept: difference between passing in txt_subdir_batches and txt_subdirs_paths
        batch_size = math.ceil(len(txt_subdirs_paths) / (os.cpu_count() // 2))
        txt_subdir_batches = []
        for i in range(0, len(txt_subdirs_paths), batch_size):
            txt_subdir_batches.append(txt_subdirs_paths[i : i + batch_size])

        ctx = get_context('spawn')
        with ctx.Pool(processes=(os.cpu_count() // 2)) as pool:
            results = pool.map(self.convert_subdirs_to_embeddings, txt_subdir_batches) # for batch
            # pool.map(self.convert_subdir_to_embeddings, txt_subdirs_paths) # not in batch i believe

            for text_batch, embed_file_path_batch in results:
                all_texts.extend(text_batch)
                all_embed_file_paths.extend(embed_file_path_batch)
        
        return all_texts, all_embed_file_paths
    
    # saves each embedding at the respective file
    def convert_embedding_to_files_batch(self, embed_and_paths):
        embed, embed_file_paths = embed_and_paths
        for output_path, embedding in zip(embed_file_paths, embed):
            file_name = output_path.replace('.txt', '.npy')
            # print(f"file_name: {file_name} has been saved.")
            np.save(file_name, embedding)
    
    # given an embedding, output each embedding into their respective embedding file paths
    def convert_embedding_to_files(self, embed, embed_file_paths):
        # split the embedding up into chunks
        chunks = np.array_split(embed, 2)
        chunk_embed_file_paths = []

        start = 0
        for i in range(len(chunks)):
            end = start + chunks[i].shape[0]
            chunk_embed_file_paths.append(embed_file_paths[start:end])
            start = end
        
        if len(chunks) != len(chunk_embed_file_paths):
            raise Exception("chunks and chunk_embed_file_paths should be the same length.")
        
        ctx = get_context('spawn')
        with ctx.Pool(processes=(2)) as pool:
            pool.map(self.convert_embedding_to_files_batch, zip(chunks, chunk_embed_file_paths)) # for batch

    # *******************************************************************************************************************
    # helper functions
    # *******************************************************************************************************************

    # makes sure that the directory specified is created
    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

# text_model should have started the process pool already
def compute_text_embeddings(text_model, model_pool, txt_path, embed_path):
    processor = TxtsToEmbeddings(txt_path, embed_path)  # note: we are not using the model in here.

    # sentences
    sentences, all_embed_file_paths = processor.convert_txts_to_embeddings()  #txts to text

    emb = text_model.model.encode_multi_process(sentences, model_pool)
    print("Embeddings computed. Shape:", emb.shape)

    # put them into embedding files 
    processor.convert_embedding_to_files(emb, all_embed_file_paths)