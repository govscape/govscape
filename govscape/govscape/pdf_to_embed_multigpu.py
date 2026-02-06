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

# for sorting file names with page numbers to ensure consistency when batching between txt and npy files (OS could 
    # order file names differently)
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def read_txt_file(txt_path):
    text = ""
    with open(txt_path, 'r') as file:
        text = file.read()
    return text 

# multiple txt subdir paths -> multiple embed dirs
# set so the number of page files matches the batch size. 
def convert_subdirs_to_embeddings(txt_path, embed_path):

    if not os.path.exists(embed_path):
        os.makedirs(embed_path)

    txt_subdirs_paths = []
    for txt_subdir in os.scandir(txt_path):
        if txt_subdir.is_dir():
            txt_subdirs_paths.append(txt_subdir.path)

    text_batch = []
    file_batch = []
    for txt_subdir_path in txt_subdirs_paths:
        embed_name = os.path.basename(txt_subdir_path)
        embedding_dir = os.path.join(embed_path, embed_name)

        if not os.path.exists(embedding_dir):
            os.makedirs(embedding_dir)

        #all txt files in the txt subdir 
        txt_files = os.listdir(txt_subdir_path)

        for txt_file in txt_files:
            txt_path = os.path.join(txt_subdir_path, txt_file)
            text = read_txt_file(txt_path)
            output_path = os.path.join(embedding_dir, txt_file)
            text_batch.append(text)
            file_batch.append(output_path)
    
    return text_batch, file_batch

# given an embedding, output each embedding into their respective embedding file paths
def convert_embedding_to_files(embeddings, embed_file_paths):

    for embedding, output_path in zip(embeddings, embed_file_paths):
        file_name = output_path.replace('.txt', '.npy')
        # print(f"file_name: {file_name} has been saved.")
        np.save(file_name, embedding)


# text_model should have started the process pool already
def compute_text_embeddings(text_model, txt_path, embed_path):
    # sentences
    sentences, all_embed_file_paths = convert_subdirs_to_embeddings(txt_path, embed_path)  #txts to text

    embeddings = text_model.encode_text_batch(sentences)
    print("Embeddings computed. Shape:", embeddings.shape)

    # put them into embedding files 
    convert_embedding_to_files(embeddings, all_embed_file_paths)