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
import pyarrow as pa
from lancedb import connect
from lancedb.query import MatchQuery, PhraseQuery
from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh.filedb.filestore import FileStorage


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
import sqlite3

class AbstractVectorIndex(ABC):

    @abstractmethod
    def __init__(self, config : IndexConfig):
        self.d = config.embedding_dim  # Assuming the embedding dimension is provided in the config
        pass

    @abstractmethod
    def build_index(self):
        pass

    @abstractmethod
    def add_batch(self, embeddings):
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
    def total_entries(self):
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

    def total_entries(self):
        return 0 # TODO: Implement this method to return the total number of embeddings in the index.

class FAISSIndex(AbstractVectorIndex):
    def __init__(self, index_directory):
        self.index_directory = index_directory
        self.faiss_index = None
        self.d = None
        self.pdf_names = []
        self.pdf_pages = []
        pass

    def add_batch(self, embeddings, pdf_names, pdf_pages):

        if self.faiss_index is None:
            self.d = embeddings.shape[1]        
            coarse_quantizer = faiss.IndexFlatL2(self.d)
            self.faiss_index = faiss.IndexIVFPQ(coarse_quantizer, self.d, 8192, int(self.d/4), 8)
            self.faiss_index.train(embeddings)
            self.faiss_index.nprobe = 32
        
        # embeddings: list or array of shape (n, d)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        if embeddings.shape[1] != self.d:
            raise ValueError(f"Embedding dimension mismatch: expected {self.d}, got {embeddings.shape[1]}")
        self.faiss_index.add(embeddings)
        self.pdf_names.extend(pdf_names)
        self.pdf_pages.extend(pdf_pages)

    def build_index(self):
        return

    def save_index(self):
        os.makedirs(self.index_directory, exist_ok=True)
        pkl.dump(self, open(self.index_directory + '/faiss_index.pkl', 'wb'))
        print(f"Index saved to {self.index_directory}/faiss_index.pkl")
        return

    def load_index(self):
        if not os.path.exists(self.index_directory + '/faiss_index.pkl'):
            return
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

    def total_entries(self):
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
    def total_entries(self):
        """
        Returns the total number of embeddings in the index.
        :return: Total number of embeddings.
        """
        pass

class LanceDBKeywordIndex(AbstractKeywordIndex):
    def __init__(self, index_keyword_directory):
        self.index_keyword_directory = index_keyword_directory
        self.db = None
        self.table = None
        self.table_name = "keyword_index"

    def _connect(self):
        if self.db is None:
            os.makedirs(self.index_keyword_directory, exist_ok=True)
            self.db = connect(self.index_keyword_directory)

    def build_index(self):
        self._connect()
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            return
        schema = pa.schema([
            pa.field("text", pa.string()),
            pa.field("pdf_name", pa.string()),
            pa.field("page", pa.int32()),
        ])
        self.table = self.db.create_table(self.table_name, schema=schema)
        self.table.create_fts_index("text", with_position=True)

    def add_batch(self, texts, pdf_names, pages):
        if self.table is None:
            self.build_index()
        rows = [
            {"text": text, "pdf_name": pdf, "page": int(page)}
            for text, pdf, page in zip(texts, pdf_names, pages)
        ]
        if rows:
            self.table.add(rows)

    def save_index(self):
        self.table.optimize()

    def load_index(self):
        self._connect()
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception as e:
            self.build_index()

    def search(self, query, k):
        if self.table is None:
            self.load_index()
        if query[0] == '"' and query[-1] == '"':
            results = (
                self.table.search(PhraseQuery(query, "text"), fts_columns="text", query_type="fts", vector_column_name='')
                .limit(k)
                .to_list()
            )
        else:
            results = (
                self.table.search(MatchQuery(query, "text"), fts_columns="text", query_type="fts", vector_column_name='')
                .limit(k)
                .to_list()
            )
        scores = [r.get("_score", 0.0) for r in results]
        pdf_names = [r["pdf_name"] for r in results]
        pages = [str(r["page"]) for r in results]
        return scores, pdf_names, pages

    def total_entries(self):
        if self.table is None:
            self.load_index()
        return self.table.count_rows()

class SQLiteKeywordIndex(AbstractKeywordIndex):
    def __init__(self, index_keyword_directory):
        self.index_keyword_directory = index_keyword_directory
        self.db_path = os.path.join(self.index_keyword_directory, "fts_txt.db")
        self.conn = None
        self.cursor = None
        self.index = None
        self._total_entries = -1
    
    def build_index(self):
        if not os.path.exists(self.index_keyword_directory):
            os.makedirs(self.index_keyword_directory)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_txt USING fts5 (text, pdf_name, page_count);                
        """)
        self.conn.commit()

    def add_batch(self, texts, pdf_names, pages):
        if self.conn is None:
            self.load_index()
        # perhaps change generate_index_keyword to build index first?
        self.cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name='fts_txt';
        """)
        exists = self.cursor.fetchone() is not None
        if not exists:
            self.build_index()
        self.cursor.execute("BEGIN TRANSACTION;")
        for text, pdf_name, page in zip(texts, pdf_names, pages):
          self.cursor.execute("INSERT INTO fts_txt (text, pdf_name, page_count) VALUES (?, ?, ?)", [text, pdf_name, page])
        self.conn.commit()

    def load_index(self):
        os.makedirs(self.index_keyword_directory, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        if self._total_entries == -1:
            self.cursor.execute("SELECT MAX(ROWID) FROM fts_txt")
            self._total_entries = self.cursor.fetchone()[0]

    def save_index(self):
        return

    def _clean_query(self, query):
        # Escape single quotes in the query
        return query.replace("-", "")
    
    def search(self, query, k):
        query = self._clean_query(query)
        if not self.cursor:
            self.load_index()
        try:
            self.cursor.execute(f'SELECT *, rank FROM fts_txt WHERE fts_txt MATCH \'{query}\' ORDER BY rank LIMIT {k}')
        except sqlite3.ProgrammingError as e:
            self.load_index()
            self.cursor.execute(f'SELECT *, rank FROM fts_txt WHERE fts_txt MATCH \'{query}\' ORDER BY rank LIMIT {k}')
        print(f'SELECT *, rank FROM fts_txt WHERE fts_txt MATCH \'{query}\' ORDER BY rank LIMIT {k}')
        distances = []
        pdf_names = []
        pages = []
        rows = self.cursor.fetchall()
        for row in rows:
            pdf_names.append(row[1])
            pages.append(str(row[2]))
            distances.append(row[3])
        return distances, pdf_names, pages    

    def total_entries(self):
        if self._total_entries == -1:
            self.load_index()
        return self._total_entries

class WhooshKeywordIndex(AbstractKeywordIndex):
    def __init__(self, index_keyword_directory):
        self.index_keyword_directory = index_keyword_directory
        self.index = None
        pass

    def build_index(self):
        schema = Schema(text=TEXT(stored=True), pdf_name=ID(stored=True), page=NUMERIC(stored=True))
        if not os.path.exists(self.index_keyword_directory):
            os.makedirs(self.index_keyword_directory)
        self.index = create_in(self.index_keyword_directory, schema)

    def add_batch(self, texts, pdf_names, pages):
        if self.index is None:
            self.build_index()
        writer = self.index.writer(procs=12, limitmb=2048)
        for text, pdf_name, page in zip(texts, pdf_names, pages):
            writer.add_document(text=text, pdf_name=pdf_name, page=page)
        writer.commit()

    def save_index(self):
        # Whoosh index is saved automatically on commit
        pass

    def load_index(self):
        if os.path.exists(self.index_keyword_directory):
            storage = FileStorage(self.index_keyword_directory, supports_mmap=False)
            self.index = storage.open_index()
        else:
            self.build_index()
    
    def search(self, query, k):
        if self.index is None:
            self.load_index()
        with self.index.searcher() as searcher:
            parser = QueryParser("text", self.index.schema)
            q = parser.parse(query)
            results = searcher.search(q, limit=k)
            pdf_names = [r['pdf_name'] for r in results]
            pages = [str(r['page']) for r in results]
            scores = [r.score for r in results]
        return scores, pdf_names, pages

    def total_entries(self):
        if self.index is None:
            self.load_index()
        with self.index.searcher() as searcher:
            return searcher.doc_count()

    def total_pages(self):
        return self.total_entries()



class AbstractMetadataIndex(ABC):

    @abstractmethod
    def __init__(self, index_metadata_directory):
        self.index_metadata_directory = index_metadata_directory
        pass

    @abstractmethod
    def build_index(self):
        """
        Instantiate the index in the directory 'self.index_metadata_directory'
        which should store the following data:
        url, crawl_date, pdf_name, sub_domain
        """
        pass

    @abstractmethod
    def add_batch(self, metadata_dicts):
        """
        Add a batch of metadata dictionaries to the index which each 
        contain url, crawl_date, pdf_name, and sub_domain.
        """
        pass

    @abstractmethod
    def load_index(self):
        """
        Load the index from 'self.index_metadata_directory'.
        """
        pass

    @abstractmethod
    def save_index(self):
        """
        Save the index to 'self.index_metadata_directory'.
        """
        pass

    @abstractmethod
    def search(self, pdf_names, filter):
        """
        Return the metadata for the pdfs in 'pdf_names' that satisfy 'filter'.
        """
        pass

    @abstractmethod
    def total_entries(self):
        """
        Returns the total number of documents in the index.
        :return: Total number of embeddings.
        """
        pass

class SQLiteMetadataIndex(AbstractMetadataIndex):
    def __init__(self, index_metadata_directory):
        self.index_metadata_directory = index_metadata_directory
        self.db_path = os.path.join(self.index_metadata_directory, "metadata.db")
        self.conn = None
        self.cursor = None
        self._total_entries = -1

    def build_index(self):
        if not os.path.exists(self.index_metadata_directory):
            os.makedirs(self.index_metadata_directory)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                crawl_url TEXT,
                crawl_date TEXT,
                pdf_name TEXT,
                sub_domain TEXT,
                page_count INTEGER,
                s3_url TEXT
            );                
        """)
        self.conn.commit()

    def add_batch(self, metadata_dicts):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        self.cursor.execute("DROP INDEX IF EXISTS idx_pdf_name;")
        self.cursor.execute("DROP INDEX IF EXISTS idx_crawl_date;")
        self.cursor.execute("DROP INDEX IF EXISTS idx_sub_domain;")
        to_insert = [
            (md.get("crawl_url", ""), md.get("crawl_date", ""), md.get("pdf_name", ""), md.get("sub_domain", ""), md.get("page_count", 0), md.get("s3_url", ""))
            for md in metadata_dicts
        ]
        self.cursor.executemany(
            "INSERT INTO metadata (crawl_url, crawl_date, pdf_name, sub_domain, page_count, s3_url) VALUES (?, ?, ?, ?, ?, ?)",
            to_insert
        )
        self.conn.commit()

    def load_index(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        if self._total_entries == -1:
            self.cursor.execute("SELECT MAX(ROWID) FROM metadata")
            self._total_entries = self.cursor.fetchone()[0]

    def save_index(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pdf_name ON metadata (pdf_name);
                            """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_crawl_date ON metadata (crawl_date);
                            """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sub_domain ON metadata (sub_domain);
                            """)
        self.conn.commit()

    def search(self, pdf_names, filter=None):
        placeholders = ",".join(f"'{name}'" for name in pdf_names)
        query = f"SELECT crawl_url, crawl_date, pdf_name, sub_domain, s3_url, page_count FROM metadata WHERE pdf_name IN ({placeholders})"
        if filter:
            for key, value in filter.items():
                if key == 'subDomain' and value != None:
                    query += f" AND sub_domain='{value}'"
                elif key == 'crawledAfter' and value != None:
                    date = value.replace("-", "") 
                    query += f" AND crawl_date>='{date}'"
                elif key == 'crawledBefore' and value != None:
                    date = value.replace("-", "") + "999999" # Pad out time to capture all times on that date
                    query += f" AND crawl_date<='{date}'"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        metadata = dict()
        for row in rows:
            pdf_name = row[2]
            row_dict = {
                    "crawl_url": row[0],
                    "crawl_date": f"{row[1][0:4]}-{row[1][4:6]}-{row[1][6:8]}",
                    "pdf_name": row[2],
                    "sub_domain": row[3],
                    "page_count": row[5]
                }
            if pdf_name not in metadata:
                metadata[pdf_name] = [row_dict]
            else:
                metadata[pdf_name].append(row_dict)
        # Return as a dict with lists of dicts representing every time the pdf was crawled
        return metadata

    def total_entries(self):
        if self._total_entries == -1:
            self.load_index()
        return self._total_entries
