# This file contains the functionality for bulk loading and indexing the data
# before requests can be served.
import random
from .config import IndexConfig
import diskannpy as dap
import numpy as np
import contextlib
import os
import pickle as pkl
import sqlite3
import sys
from abc import ABC, abstractmethod

import numpy as np

import diskannpy as dap
import pyarrow as pa
from lancedb import connect
from lancedb.query import MatchQuery, PhraseQuery
from whoosh.fields import ID, NUMERIC, TEXT, Schema
from whoosh.filedb.filestore import FileStorage
import meilisearch
from typing import Optional, Sequence

# Optional Elasticsearch client (not required for other backends)
_ES_AVAILABLE = True
try:  # pragma: no cover - optional dependency
    from elasticsearch import Elasticsearch
    from elasticsearch import helpers as es_helpers
except Exception:
    _ES_AVAILABLE = False

lucene = None
_LUCENE_LOADED = False

def _load_lucene():
    global lucene, _LUCENE_LOADED
    if _LUCENE_LOADED:
        return

    import lucene as _lucene
    lucene = _lucene
    lucene.initVM()

    # import Java-side classes ONLY after initVM
    global Paths, FSDirectory, StandardAnalyzer, Document, Field, StringField, TextField, StoredField
    global IndexWriter, IndexWriterConfig, DirectoryReader, IndexSearcher, QueryParser, BM25Similarity

    from java.nio.file import Paths
    from org.apache.lucene.store import FSDirectory
    from org.apache.lucene.analysis.standard import StandardAnalyzer
    from org.apache.lucene.document import Document, Field, StringField, TextField, StoredField
    from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader
    from org.apache.lucene.search import IndexSearcher
    from org.apache.lucene.queryparser.classic import QueryParser
    from org.apache.lucene.search.similarities import BM25Similarity
from whoosh.index import create_in
from whoosh.qparser import QueryParser

    _LUCENE_LOADED = True

# Avoid annoying output from faiss during import
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
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
    def __init__(self):
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

    @abstractmethod
    def total_entries(self):
        """
        Returns the total number of embeddings in the index.
        :return: Total number of embeddings.
        """


class DiskANNIndex(AbstractVectorIndex):
    def __init__(self, embedding_directory, index_directory):
        self.embedding_directory = embedding_directory
        self.index_directory = index_directory
        self.index = None
        self.page_indices = None

    def build_index(self):
        if not os.path.exists(self.index_directory):
            os.makedirs(self.index_directory)
        embedding = os.path.join(self.embedding_directory, "embeddings.bin")
        # Comments below are adapted from the DiskANN repo.
        # Can also be cosine, especially if you don't normalize your vectors.
        # Higher complexity considers more candidate points when ranking.
        # Higher graph degree increases quality but takes longer to build.
        # Memory values are in GB.
        # 0 means use all available threads.
        # ann is the default prefix; files are generated like f"{index_prefix}_".
        # Product quantization can improve recall at lower latency; skip for now.
        dap.build_disk_index(
            data=embedding,
            distance_metric="l2",
            index_directory=self.index_directory,
            complexity=128,
            graph_degree=64,
            search_memory_maximum=16.0,
            build_memory_maximum=100.0,
            num_threads=0,
            vector_dtype=np.float32,
            index_prefix="ann",
            pq_disk_bytes=0,
        )

    def save_index(self, filepath):
        return

    def load_index(self):
        self.index = dap.StaticDiskIndex(
            index_directory=self.index_directory,
            num_threads=0,
            num_nodes_to_cache=17,
            index_prefix="ann",
        )

    def search(self, query_vector, k):
        query_vector = query_vector.copy() / np.linalg.norm(query_vector)
        # query vector should be 2D
        internal_indices, distances = self.index.search(
            query=query_vector,
            k_neighbors=k,
            complexity=k * 10,  # must be as big or bigger than `k_neighbors`
        )
        return distances, internal_indices

    def total_entries(self):
        # TODO: Implement this method to return the total number of embeddings.
        return 0


class FAISSIndex(AbstractVectorIndex):
    def __init__(self, index_directory):
        self.index_directory = index_directory
        self.faiss_index = None
        self.d = None
        self.is_trained = False
        self.train_batch = None
        self.pdf_names = []
        self.pdf_pages = []
        faiss.omp_set_num_threads(os.cpu_count())

    def add_batch(self, embeddings, pdf_names, pdf_pages):
        # embeddings: list or array of shape (n, d)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        if self.d is None:
            self.d = embeddings.shape[1]
            self.train_batch = np.array([], dtype=np.float32).reshape(0, self.d)

        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatL2(self.d)

        if not self.is_trained:
            self.train_batch = np.vstack((self.train_batch, embeddings))
            if self.train_batch.shape[0] >= 8192 * 39:
                coarse_quantizer = faiss.IndexFlatL2(self.d)
                self.faiss_index = faiss.IndexIVFPQ(
                    coarse_quantizer, self.d, 8192, int(self.d / 4), 8
                )
                self.faiss_index.train(self.train_batch)
                self.faiss_index.nprobe = 32
                self.faiss_index.add(
                    self.train_batch[: -embeddings.shape[0], :]
                )  # add all but the last batch
                self.is_trained = True

        if embeddings.shape[1] != self.d:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.d}, got {embeddings.shape[1]}"
            )
        self.faiss_index.add(embeddings)
        self.pdf_names.extend(pdf_names)
        self.pdf_pages.extend(pdf_pages)

    def build_index(self):
        return

    def save_index(self):
        os.makedirs(self.index_directory, exist_ok=True)
        index_path = os.path.join(self.index_directory, "faiss_index.pkl")
        with open(index_path, "wb") as handle:
            pkl.dump(self, handle)
        print(f"Index saved to {self.index_directory}/faiss_index.pkl")
        return

    def load_index(self):
        index_path = os.path.join(self.index_directory, "faiss_index.pkl")
        if not os.path.exists(index_path):
            return
        with open(index_path, "rb") as handle:
            index = pkl.load(handle)
        self.faiss_index = index.faiss_index
        self.train_batch = index.train_batch
        self.is_trained = index.is_trained
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
        distances, indices = self.faiss_index.search(query_embedding, k)
        distances = distances[0]
        indices = indices[0]
        name_results = []
        page_results = []
        for i in range(indices.shape[0]):
            # parse file information for page
            pdf_name = self.pdf_names[indices[i]]
            pdf_page = self.pdf_pages[indices[i]]
            name_results.append(pdf_name)
            page_results.append(pdf_page)
        return distances, name_results, page_results

    def total_entries(self):
        return self.faiss_index.ntotal


class AbstractKeywordIndex(ABC):
    @abstractmethod
    def __init__(self, index_keyword_directory):
        self.index_keyword_directory = index_keyword_directory

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

    @abstractmethod
    def total_entries(self):
        """
        Returns the total number of embeddings in the index.
        :return: Total number of embeddings.
        """


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
        schema = pa.schema(
            [
                pa.field("text", pa.string()),
                pa.field("pdf_name", pa.string()),
                pa.field("page", pa.int32()),
            ]
        )
        self.table = self.db.create_table(self.table_name, schema=schema)
        self.table.create_fts_index("text", with_position=True)

    def add_batch(self, texts, pdf_names, pages):
        if self.table is None:
            self.build_index()
        rows = [
            {"text": text, "pdf_name": pdf, "page": int(page)}
            for text, pdf, page in zip(texts, pdf_names, pages, strict=False)
        ]
        if rows:
            self.table.add(rows)

    def save_index(self):
        self.table.optimize()

    def load_index(self):
        self._connect()
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            self.build_index()

    def search(self, query, k):
        if self.table is None:
            self.load_index()
        if query[0] == '"' and query[-1] == '"':
            results = (
                self.table.search(
                    PhraseQuery(query, "text"),
                    fts_columns="text",
                    query_type="fts",
                    vector_column_name="",
                )
                .limit(k)
                .to_list()
            )
        else:
            results = (
                self.table.search(
                    MatchQuery(query, "text"),
                    fts_columns="text",
                    query_type="fts",
                    vector_column_name="",
                )
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
        self.VALID_CHARS = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _"
        )

    def build_index(self):
        if not os.path.exists(self.index_keyword_directory):
            os.makedirs(self.index_keyword_directory)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_txt USING fts5 (
                text,
                pdf_name,
                page_count
            );
            """
        )
        self.conn.commit()

    def add_batch(self, texts, pdf_names, pages):
        if self.conn is None:
            self.load_index()
        self.cursor.execute("BEGIN TRANSACTION;")
        for text, pdf_name, page in zip(texts, pdf_names, pages, strict=False):
            self.cursor.execute(
                "INSERT INTO fts_txt (text, pdf_name, page_count) VALUES (?, ?, ?)",
                [text, pdf_name, page],
            )
        self.conn.commit()

    def load_index(self):
        os.makedirs(self.index_keyword_directory, exist_ok=True)
        if not os.path.exists(self.db_path):
            self.build_index()
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        if self._total_entries == -1:
            self.cursor.execute("SELECT MAX(ROWID) FROM fts_txt")
            self._total_entries = self.cursor.fetchone()[0]

    def save_index(self):
        return

    def _valid_char(self, c):
        return c in self.VALID_CHARS or not c.isascii()

    # SQLITE FTS5 requires special handling of punctuation and quotes
    def _clean_query(self, query):
        in_quote = False
        new_string = ""
        for c in query:
            if not in_quote and c == '"':
                in_quote = True
                new_string += c
                continue
            if in_quote and c == '"':
                in_quote = False
                new_string += c
                continue
            if in_quote:
                new_string += c
                continue
            if not in_quote and self._valid_char(c):
                new_string += c
            else:
                new_string += " "
        if in_quote:
            new_string += '"'
        return new_string

    def search(self, query, k):
        query = self._clean_query(query)
        if not self.cursor:
            self.load_index()
        search_query = (
            "SELECT *, rank FROM fts_txt WHERE fts_txt MATCH ? ORDER BY rank LIMIT ?"
        )
        try:
            self.cursor.execute(search_query, (query, k))
        except sqlite3.ProgrammingError:
            self.load_index()
            self.cursor.execute(search_query, (query, k))
        distances = []
        pdf_names = []
        pages = []
        rows = self.cursor.execute(search_query, (query, k)).fetchall()
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

    def build_index(self):
        schema = Schema(
            text=TEXT(stored=True), pdf_name=ID(stored=True), page=NUMERIC(stored=True)
        )
        if not os.path.exists(self.index_keyword_directory):
            os.makedirs(self.index_keyword_directory)
        self.index = create_in(self.index_keyword_directory, schema)

    def add_batch(self, texts, pdf_names, pages):
        if self.index is None:
            self.build_index()
        # Whoosh's multi-process writer is prone to finish_subsegment races when the
        # caller exits or crashes early, so stick with a single-process writer for
        # stability.
        writer = self.index.writer(procs=12, limitmb=512)
        for text, pdf_name, page in zip(texts, pdf_names, pages, strict=False):
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
            pdf_names = [r["pdf_name"] for r in results]
            pages = [str(r["page"]) for r in results]
            scores = [r.score for r in results]
        return scores, pdf_names, pages

    def total_entries(self):
        if self.index is None:
            self.load_index()
        with self.index.searcher() as searcher:
            return searcher.doc_count()

    def total_pages(self):
        return self.total_entries()


class LuceneKeywordIndex(AbstractKeywordIndex):
    def __init__(self, index_keyword_directory):
        _load_lucene()
        self.index_keyword_directory = index_keyword_directory
        os.makedirs(self.index_keyword_directory, exist_ok=True)

        self._dir = None
        self._analyzer = None
        self._writer = None

        self._reader = None
        self._searcher = None 

    def _attach(self):
        """
        Synchronizes lucene with the current python thread
        """
        _load_lucene()
        env = lucene.getVMEnv()
        if env is not None:
            env.attachCurrentThread()

    def _open_dir(self):
        if self._dir is None:
            self._dir = FSDirectory.open(Paths.get(self.index_keyword_directory))

    def build_index(self):
        """
        Create/open the index for writing.
        """
        self._attach()
        self._open_dir()
        if self._analyzer is None:
            self._analyzer = StandardAnalyzer()

        if self._writer is None:
            cfg = IndexWriterConfig(self._analyzer)
            cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
            self._writer = IndexWriter(self._dir, cfg)

    def add_batch(self, texts, pdf_names, pages):
        """
        Adds documents to the Lucene index.
        """
        self._attach()
        if self._writer is None:
            self.build_index()

        # If we've opened a reader/searcher, it won't see newly added docs until refreshed.
        for text, pdf_name, page in zip(texts, pdf_names, pages):
            doc = Document()

            # Indexed full-text field
            doc.add(TextField("text", text if text is not None else "", Field.Store.NO))

            # Stored fields to return in results
            doc.add(StringField("pdf_name", pdf_name if pdf_name is not None else "", Field.Store.YES))

            doc.add(StoredField("page", int(page)))

            self._writer.addDocument(doc)

    def save_index(self):
        """
        Commit changes (and optionally close writer).
        """
        self._attach()
        if self._writer is not None:
            self._writer.commit()
            # You can keep it open for more batches; closing is safer for "bulk build then serve".
            self._writer.close()
            self._writer = None

        # Searcher must be refreshed after commits.
        self._close_reader()

    def _close_reader(self):
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
        self._reader = None
        self._searcher = None

    def load_index(self):
        """
        Open the index for searching.
        """
        self._attach()
        self._open_dir()
        if self._analyzer is None:
            self._analyzer = StandardAnalyzer()

        # Close any existing reader so we don't leak file handles.
        self._close_reader()

        # DirectoryReader.open will fail if index doesn't exist yet.
        try:
            self._reader = DirectoryReader.open(self._dir)
            self._searcher = IndexSearcher(self._reader)
            self._searcher.setSimilarity(BM25Similarity())
        except Exception:
            # No index yet; create it.
            self.build_index()
            self.save_index()
            self._reader = DirectoryReader.open(self._dir)
            self._searcher = IndexSearcher(self._reader)
            self._searcher.setSimilarity(BM25Similarity())

    def _ensure_searcher_fresh(self):
        """
        If writer exists and has uncommitted changes, commit them;
        then open/refresh reader so searches see newest docs.
        """
        self._attach()
        if self._writer is not None:
            self._writer.commit()

        if self._reader is None or self._searcher is None:
            self.load_index()
            return

        # Try to reopen reader if index changed.
        new_reader = DirectoryReader.openIfChanged(self._reader)
        if new_reader is not None:
            self._reader.close()
            self._reader = new_reader
            self._searcher = IndexSearcher(self._reader)

    def search(self, query, k):
        self._attach()
        self._ensure_searcher_fresh()

        parser = QueryParser("text", self._analyzer)
        try:
            q = parser.parse(query)
        except Exception:
            q = parser.parse(QueryParser.escape(query))

        top_docs = self._searcher.search(q, int(k))
        hits = top_docs.scoreDocs

        stored = self._reader.storedFields()

        scores, pdf_names, pages = [], [], []
        for sd in hits:
            doc = stored.document(sd.doc)
            scores.append(float(sd.score))
            pdf_names.append(doc.get("pdf_name"))
            pages.append(str(doc.get("page")))

        return scores, pdf_names, pages


    def total_entries(self):
        self._attach()
        if self._reader is None:
            self.load_index()
        return int(self._reader.numDocs())


class ElasticsearchKeywordIndex(AbstractKeywordIndex):
    def __init__(self, index_keyword_directory):
        if not _ES_AVAILABLE:
            raise RuntimeError("Elasticsearch client not available.")
        self.index_keyword_directory = index_keyword_directory
        os.makedirs(self.index_keyword_directory, exist_ok=True)

        self.hosts = ["http://host.docker.internal:9200"]

        self.index_name = "govscape_keyword"
        self._es: Optional[Elasticsearch] = None

    def _client(self) -> Elasticsearch:
        if self._es is None:
            self._es = Elasticsearch(self.hosts, api_key="")
        return self._es

    def build_index(self):
        es = self._client()
        try:
            exists = es.indices.exists(index=self.index_name)
        except Exception:
            # Some client versions return bool directly, others return dict.
            exists = False
        if exists:
            return
        mapping = {
            "settings": {
                "index": {
                    # Faster visibility for tests; adjust in production.
                    "refresh_interval": "1s"
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "pdf_name": {"type": "keyword"},
                    "page": {"type": "integer"},
                }
            },
        }
        es.indices.create(index=self.index_name, **mapping)

    def add_batch(self, texts, pdf_names, pages):
        es = self._client()
        # Ensure index exists before bulk.
        self.build_index()
        actions = (
            {
                "_index": self.index_name,
                "_source": {
                    "text": text if text is not None else "",
                    "pdf_name": pdf if pdf is not None else "",
                    "page": int(page),
                },
            }
            for text, pdf, page in zip(texts, pdf_names, pages)
        )
        es_helpers.bulk(es, actions)

    def save_index(self):
        # Force refresh so subsequent searches see newly indexed docs.
        es = self._client()
        try:
            es.indices.refresh(index=self.index_name)
        except Exception:
            pass

    def load_index(self):
        self._client()
        self.build_index()

    def search(self, query, k):
        es = self._client()

        is_phrase = len(query) >= 2 and query[0] == '"' and query[-1] == '"'
        q_body = {
            "size": int(k),
            "query": {
                "match_phrase" if is_phrase else "match": {
                    "text": query.strip('"') if is_phrase else query
                }
            }
        }
        resp = es.search(index=self.index_name, body=q_body)
        hits = resp.get("hits", {}).get("hits", [])
        scores = [float(h.get("_score", 0.0)) for h in hits]
        pdf_names = [h.get("_source", {}).get("pdf_name", "") for h in hits]
        pages = [str(h.get("_source", {}).get("page", "")) for h in hits]
        return scores, pdf_names, pages

    def total_entries(self):
        es = self._client()
        try:
            resp = es.count(index=self.index_name)
            return int(resp.get("count", 0))
        except Exception:
            return 0


class MeilisearchKeywordIndex(AbstractKeywordIndex):
    def __init__(self, index_keyword_directory):
        print("Connecting to Meilisearch at http://host.docker.internal:7700/")
        self._client = meilisearch.Client("http://host.docker.internal:7700/", "masterKey")

        self._index = self._client.index("govscape_keyword")


    def build_index(self):
        pass

    def add_batch(self, texts, pdf_names, pages):
        print(f"Indexing batch of {len(texts)} documents into Meilisearch...")
        documents = [
            {
                # Id will be a random int for this test
                "id" : str(random.randint(1, 1_000_000)),
                "text": text if text is not None else "",
                "pdf_name": pdf if pdf is not None else "",
                "page": int(page),
            }
            for text, pdf, page in zip(texts, pdf_names, pages)
        ]
        tasks = self._index.add_documents_in_batches(documents)

        for task in tasks:
            self._client.wait_for_task(task.task_uid, timeout_in_ms=300000)

        failed = self._client.get_tasks({"statuses": ["failed"], "limit": 20})
        print(failed)

    def save_index(self):
        pass

    def load_index(self):
        pass

    def search(self, query, k):
        results = self._index.search(query, {"limit": k})
        hits = results.get("hits", [])
        scores = [float(h.get("_rankingScore", 0.0)) for h in hits]
        pdf_names = [h.get("pdf_name", "") for h in hits]
        pages = [str(h.get("page", "")) for h in hits]
        return scores, pdf_names, pages

    def total_entries(self):
        stats = self._index.get_stats()
        return stats.number_of_documents

class AbstractMetadataIndex(ABC):
    @abstractmethod
    def __init__(self, index_metadata_directory):
        self.index_metadata_directory = index_metadata_directory

    @abstractmethod
    def build_index(self):
        """
        Instantiate the index in the directory 'self.index_metadata_directory'
        which should store the following data:
        url, crawl_date, pdf_name, sub_domain
        """

    @abstractmethod
    def add_batch(self, metadata_dicts):
        """
        Add a batch of metadata dictionaries to the index which each
        contain url, crawl_date, pdf_name, and sub_domain.
        """

    @abstractmethod
    def load_index(self):
        """
        Load the index from 'self.index_metadata_directory'.
        """

    @abstractmethod
    def save_index(self):
        """
        Save the index to 'self.index_metadata_directory'.
        """

    @abstractmethod
    def search(self, pdf_names, filter):
        """
        Return the metadata for the pdfs in 'pdf_names' that satisfy 'filter'.
        """

    @abstractmethod
    def total_entries(self):
        """
        Returns the total number of documents in the index.
        :return: Total number of embeddings.
        """


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
            (
                md.get("crawl_url", ""),
                md.get("crawl_date", ""),
                md.get("pdf_name", ""),
                md.get("sub_domain", ""),
                md.get("page_count", 0),
                md.get("s3_url", ""),
            )
            for md in metadata_dicts
        ]
        self.cursor.executemany(
            "INSERT INTO metadata ("
            "crawl_url, crawl_date, pdf_name, sub_domain, page_count, s3_url"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            to_insert,
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
        query = (
            "SELECT crawl_url, crawl_date, pdf_name, sub_domain, s3_url, page_count "
            f"FROM metadata WHERE pdf_name IN ({placeholders})"
        )
        if filter:
            for key, value in filter.items():
                if key == "sub_domain" and value is not None:
                    query += f" AND sub_domain='{value}'"
                elif key == "crawled_after" and value is not None:
                    date = value.replace("-", "")
                    query += f" AND crawl_date>='{date}'"
                elif key == "crawled_before" and value is not None:
                    date = (
                        value.replace("-", "") + "999999"
                    )  # Pad out time to capture all times on that date
                    query += f" AND crawl_date<='{date}'"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        metadata = {}
        for row in rows:
            pdf_name = row[2]
            row_dict = {
                "crawl_url": row[0],
                "crawl_date": f"{row[1][0:4]}-{row[1][4:6]}-{row[1][6:8]}",
                "pdf_name": row[2],
                "sub_domain": row[3],
                "page_count": row[5],
            }
            if pdf_name not in metadata:
                metadata[pdf_name] = [row_dict]
            else:
                metadata[pdf_name].append(row_dict)
        # Return as a dict with lists of dicts representing every time the pdf was
        # crawled.
        return metadata

    def total_entries(self):
        if self._total_entries == -1:
            self.load_index()
        return self._total_entries
