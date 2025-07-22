# This file contains the functionality for bulk loading and indexing the data 
# before requests can be served.
from .config import IndexConfig
import diskannpy as dap
import numpy as np
import os
import pickle as pkl
from abc import ABC, abstractmethod
import contextlib
import sys
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser

# Avoid annoying output from faiss during import
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

with suppress_output():
    import faiss

class AbstractVectorIndex(ABC):

    @abstractmethod
    def __init__(self, config : IndexConfig):
        self.d = config.embedding_dim  # Assuming the embedding dimension is provided in the config
        pass

    @abstractmethod
    def build_index(self):
        pass

    @abstractmethod
    def load_index(self):
        pass

    @abstractmethod
    def save_index(self):
        pass
    
    @abstractmethod
    def search(self, query_vector, k):
        """
        Search for the k closest PDFs to the query vector.
        :param query_vector: The vector to search for.
        :param k: The number of closest arrays to return.
        :return: A tuple of distances, pdf_names, and pages.
        """
        pass

    @abstractmethod
    def total_embeddings(self):
        """
        Returns the total number of embeddings in the index.
        :return: Total number of embeddings.
        """
        pass

class DiskANNIndex(AbstractVectorIndex):
    def __init__(self, embedding_directory, index_directory):
        self.embedding_directory = embedding_directory
        self.index_directory = index_directory
        self.index = None
        self.page_indices = None
        pass

    def build_index(self):
        if not os.path.exists(self.index_directory):
            os.makedirs(self.index_directory)
        embedding = os.path.join(self.embedding_directory, "embeddings.bin")
        dap.build_disk_index( # comments from DiskANN repo
            data=embedding,
            distance_metric="l2", # can also be cosine, especially if you don't normalize your vectors like above
            index_directory=self.index_directory, 
            complexity=128,  # the larger this is, the more candidate points we consider when ranking
            graph_degree=64,  # the beauty of a vamana index is it's ability to shard and be able to transfer long distances across the grpah without navigating the whole thing. the larger this value is, the higher quality your results, but the longer it will take to build 
            search_memory_maximum=16.0, # a floating point number to represent how much memory in GB we want to optimize for @ query time
            build_memory_maximum=100.0, # a floating point number to represent how much memory in GB we are allocating for the index building process
            num_threads=0,  # 0 means use all available threads - but if you are in a shared environment you may need to restrict how greedy you are
            vector_dtype=np.float32, 
            index_prefix="ann",  # ann is the default anyway. all files generated will have the prefix `ann_`, in the form of `f"{index_prefix}_"`
            pq_disk_bytes=0  # using product quantization of your vectors can still achieve excellent recall characteristics at a fraction of the latency, but we'll do it without PQ for now
        )

    def save_index(self, filepath):
        return

    def load_index(self):
        self.index = dap.StaticDiskIndex(
            index_directory=self.index_directory,
            num_threads=0,
            num_nodes_to_cache=17,
            index_prefix="ann"  
        )

    def search(self, query_vector, k):
        query_vector = query_vector.copy() / np.linalg.norm(query_vector)
        # query vector should be 2D
        internal_indices, distances = self.index.search(
            query=query_vector,
            k_neighbors=k,
            complexity=k*10,  # must be as big or bigger than `k_neighbors`
        )
        return distances, internal_indices

    def total_embeddings(self):
        return 0 # TODO: Implement this method to return the total number of embeddings in the index.

class FAISSIndex(AbstractVectorIndex):
    def __init__(self, embedding_directory, index_directory):
        self.embedding_directory = embedding_directory
        self.index_directory = index_directory
        self.faiss_index = None
        self.pdf_names = []
        self.pdf_pages = []
        pass

    def build_index(self):
        # Train model on test vectors
        self.pdf_names = []
        self.pdf_pages = []
        npy_files = []
        for root, _, files in os.walk(self.embedding_directory):
            for file in files:
                if file.endswith(".npy"):
                    npy_files.append(os.path.join(root, file))
                    file = os.path.splitext(file)[0]  # remove .npy extension
                    self.pdf_names.append(file.rpartition('_')[0])
                    self.pdf_pages.append(file.rpartition('_')[2])

        # Load each .npy file into an array
        stacked_array = np.vstack([np.load(file) for file in npy_files])
        self.d = stacked_array.shape[1]

        # construct faiss index index
        self.faiss_index = faiss.IndexFlatL2(self.d)
        self.faiss_index.add(stacked_array)
        if not os.path.exists(self.index_directory):
            os.makedirs(self.index_directory)
        

    def save_index(self):
        pkl.dump(self, open(self.index_directory + '/faiss_index.pkl', 'wb'))
        print(f"Index saved to {self.index_directory}/faiss_index.pkl")
        return

    def load_index(self):
        index = pkl.load(open(self.index_directory + '/faiss_index.pkl', 'rb'))
        self.faiss_index = index.faiss_index
        self.pdf_names = index.pdf_names
        self.pdf_pages = index.pdf_pages
        self.d = index.faiss_index.d
        print(f"Index loaded from {self.index_directory}/faiss_index.pkl")
        return

    def search(self, query_embedding, k):
        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis, :]
        if query_embedding.ndim != 2:
            raise ValueError("Query embedding must be a 2D array.")
        D, I = self.faiss_index.search(query_embedding, k)
        D = D[0]
        I = I[0]
        name_results = []
        page_results = []
        for i in range(I.shape[0]):
            # parse file information for page
            pdf_name = self.pdf_names[I[i]]
            pdf_page = self.pdf_pages[I[i]]
            name_results.append(pdf_name)
            page_results.append(pdf_page)
        return D, name_results, page_results

    def total_embeddings(self):
        return self.faiss_index.ntotal


class AbstractKeywordIndex(ABC):

    @abstractmethod
    def __init__(self, index_keyword_directory):
        self.index_keyword_directory = index_keyword_directory
        pass

    @abstractmethod
    def build_index(self):
        pass

    @abstractmethod
    def add_batch(self, texts, pdf_names, pages):
        pass

    @abstractmethod
    def load_index(self):
        pass

    @abstractmethod
    def save_index(self):
        pass

    @abstractmethod
    def search(self, query_vector, k):
        """
        Search for the k closest PDFs to the query vector.
        :param query_vector: The vector to search for.
        :param k: The number of closest arrays to return.
        :return: A tuple of distances, pdf_names, and pages.
        """
        pass

    @abstractmethod
    def total_texts(self):
        """
        Returns the total number of embeddings in the index.
        :return: Total number of embeddings.
        """
        pass

class WhooshIndex(AbstractKeywordIndex):
    def __init__(self, index_keyword_directory):
        self.index_keyword_directory = index_keyword_directory
        self.index = None
        pass

    def build_index(self):
        # Define schema: text (content), pdf_name, page
        schema = Schema(text=TEXT(stored=True), pdf_name=ID(stored=True), page=NUMERIC(stored=True))
        if not os.path.exists(self.index_keyword_directory):
            os.makedirs(self.index_keyword_directory)
        self.index = create_in(self.index_keyword_directory, schema)

    # If the index does not exist, call build_index to create it
    # Then, add the documents to the index
    def add_batch(self, texts, pdf_names, pages):
        # If index doesn't exist, build it
        if self.index is None:
            self.build_index()
        writer = self.index.writer()
        for text, pdf_name, page in zip(texts, pdf_names, pages):
            writer.add_document(text=text, pdf_name=pdf_name, page=page)
        writer.commit()

    def save_index(self):
        # Whoosh index is saved automatically on commit
        pass

    def load_index(self):
        if os.path.exists(self.index_keyword_directory):
            self.index = open_dir(self.index_keyword_directory)
        else:
            raise FileNotFoundError(f"Index directory {self.index_keyword_directory} does not exist.")

    def search(self, query, k):
        if self.index is None:
            self.load_index()
        with self.index.searcher() as searcher:
            parser = QueryParser("text", self.index.schema)
            q = parser.parse(query)
            results = searcher.search(q, limit=k)
            pdf_names = [r['pdf_name'] for r in results]
            pages = [r['page'] for r in results]
            scores = [r.score for r in results]
        return scores, pdf_names, pages

    def total_texts(self):
        if self.index is None:
            self.load_index()
        with self.index.searcher() as searcher:
            return searcher.doc_count()

    def total_pages(self):
        # Alias for total_texts (since each doc is a page)
        return self.total_texts()