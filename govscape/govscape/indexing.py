# AI modified: 2026-02-21 d8ae3e4a
# AI modified: 2026-03-01 b6756050
# AI modified: 2026-03-08 f62d40b8
# AI modified: 2026-03-09 4efba197
# AI modified: 2026-03-09 4efba197
# AI modified: 2026-03-14 21:55:15 1c688b19
# AI modified: 2026-03-14 22:34:51 1c688b19
# AI modified: 2026-03-14 22:38:50 1c688b19
# AI modified: 2026-03-15 03:26:30 1c688b19
# AI modified: 2026-04-06 00:10:53 434ce298
# AI modified: 2026-04-12 18:24:13 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 18:24:13 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 22:41:29 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 22:50:07 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 22:50:07 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 23:02:34 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 23:05:04 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 23:40:59 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 23:42:48 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 23:49:00 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-12 23:51:04 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# AI modified: 2026-04-13 00:58:18 fddc6344a807a84c8b9161bd3ffeded5153c5e27
# This file contains the functionality for bulk loading and indexing the data
# before requests can be served.
import contextlib
import os
import pickle as pkl
import sqlite3
import sys
import threading
from abc import ABC, abstractmethod

import numpy as np

import duckdb
import pyarrow as pa
from lancedb import connect
from lancedb.query import MatchQuery, PhraseQuery
from whoosh.fields import ID, NUMERIC, TEXT, Schema
from whoosh.filedb.filestore import FileStorage
from whoosh.index import create_in
from whoosh.qparser import QueryParser

from .query import EqualityPredicate, Predicate, RangePredicate

lucene = None
_LUCENE_LOADED = False
# Prevents two threads from racing through initVM() simultaneously.
_lucene_load_lock = threading.Lock()


# Lazily load lucene and its Java dependencies only when needed, since it may
# not exist in all environments.
def _load_lucene():
    global lucene, _LUCENE_LOADED
    # Already loaded (no lock needed for read).
    if _LUCENE_LOADED:
        return
    # Module lock to prevent two threads both calling initVM().
    with _lucene_load_lock:
        if _LUCENE_LOADED:
            return
        try:
            import lucene as _lucene

            lucene = _lucene
            lucene.initVM()

            # import Java-side classes ONLY after initVM
            global \
                Paths, \
                FSDirectory, \
                StandardAnalyzer, \
                Document, \
                Field, \
                StringField, \
                TextField, \
                StoredField
            global \
                IndexWriter, \
                IndexWriterConfig, \
                DirectoryReader, \
                IndexSearcher, \
                QueryParser, \
                BM25Similarity

            from java.nio.file import Paths  # type: ignore[import]
            from org.apache.lucene.analysis.standard import (  # type: ignore[import]
                StandardAnalyzer,
            )
            from org.apache.lucene.document import (  # type: ignore[import]
                Document,
                Field,
                StoredField,
                StringField,
                TextField,
            )
            from org.apache.lucene.index import (  # type: ignore[import]
                DirectoryReader,
                IndexWriter,
                IndexWriterConfig,
            )
            from org.apache.lucene.queryparser.classic import (  # type: ignore[import]
                QueryParser,
            )
            from org.apache.lucene.search import IndexSearcher  # type: ignore[import]
            from org.apache.lucene.search.similarities import (  # type: ignore[import]
                BM25Similarity,
            )
            from org.apache.lucene.store import FSDirectory  # type: ignore[import]

            _LUCENE_LOADED = True
        except Exception as e:
            print(f"Lucene not available: {e}")
            raise ImportError("Lucene is not available in this environment") from e


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

    @abstractmethod
    def get_vectors_for_pdf_page_counts(self, pdf_page_counts):
        """
        Return a mapping: {pdf_name: [(page, embedding_vector), ...]} for pages
        where page < pdf_page_counts[pdf_name].
        """


class FAISSIndex(AbstractVectorIndex):
    def __init__(self, index_directory, index_type="IVFPQ"):
        self.index_directory = index_directory
        self.faiss_index = None
        self.index_type = index_type
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
                if self.index_type == "IVFPQ":
                    coarse_quantizer = faiss.IndexFlatL2(self.d)
                    self.faiss_index = faiss.IndexIVFPQ(
                        coarse_quantizer, self.d, 8192, int(self.d / 4), 8
                    )
                elif self.index_type == "Flat":
                    self.faiss_index = faiss.IndexFlatL2(self.d)
                elif self.index_type == "IVF":
                    quantizer = faiss.IndexFlatL2(
                        self.d
                    )  # coarse quantizer used for IVF (inverted file) indexing
                    self.faiss_index = faiss.IndexIVFFlat(
                        quantizer, self.d, 8192, faiss.METRIC_L2
                    )
                elif self.index_type == "HNSW":
                    self.faiss_index = faiss.IndexHNSWFlat(self.d, 32)
                    self.faiss_index.hnsw.efConstruction = 200
                    self.faiss_index.hnsw.efSearch = 200
                else:
                    raise ValueError(f"Unsupported index type: {self.index_type}")
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

    def get_vectors_for_pdf_page_counts(self, pdf_page_counts):
        if not pdf_page_counts:
            return {}

        candidates = {}
        for idx, (pdf_name, pdf_page) in enumerate(
            zip(self.pdf_names, self.pdf_pages, strict=False)
        ):
            max_pages = pdf_page_counts.get(pdf_name)
            if max_pages is None:
                continue
            try:
                page_num = int(pdf_page)
            except (TypeError, ValueError):
                continue
            if page_num < 0 or page_num >= int(max_pages):
                continue
            try:
                vector = np.asarray(
                    self.faiss_index.reconstruct(idx),
                    dtype=np.float32,
                )
            except Exception:
                continue
            if pdf_name not in candidates:
                candidates[pdf_name] = []
            candidates[pdf_name].append((str(pdf_page), vector))
        return candidates


class LanceDBVectorIndex(AbstractVectorIndex):
    def __init__(self, index_directory, table_name="vector_index"):
        self.index_directory = index_directory
        self.table_name = table_name
        self.db = None
        self.table = None
        self.d = None

    def _connect(self):
        if self.db is None:
            os.makedirs(self.index_directory, exist_ok=True)
            self.db = connect(self.index_directory)

    def build_index(self):
        self._connect()
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            return

        if self.d is None:
            raise ValueError(
                "Vector dimension must be known before creating LanceDB vector table"
            )

        if self.table_name not in self.db.table_names():
            schema = pa.schema(
                [
                    pa.field("vector", pa.list_(pa.float32(), self.d)),
                    pa.field("pdf_name", pa.string()),
                    pa.field("page", pa.int32()),
                ]
            )
            self.table = self.db.create_table(self.table_name, schema=schema)

    def add_batch(self, embeddings, pdf_names, pdf_pages):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if self.d is None:
            self.d = embeddings.shape[1]
        elif embeddings.shape[1] != self.d:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.d}, got {embeddings.shape[1]}"
            )

        if self.table is None:
            self.build_index()

        rows = [
            {
                "vector": embeddings[i].tolist(),
                "pdf_name": str(pdf_name),
                "page": int(page),
            }
            for i, (pdf_name, page) in enumerate(
                zip(pdf_names, pdf_pages, strict=False)
            )
        ]
        if rows:
            self.table.add(rows)

    def save_index(self):
        if self.table is None:
            self.load_index()
        if self.table is None:
            return

        if self.table.count_rows() > 1024:
            try:
                self.table.create_index(vector_column_name="vector")
            except Exception as exc:
                if "already exists" not in str(exc).lower():
                    raise
        self.table.optimize()

    def load_index(self):
        self._connect()
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            self.build_index()
        if self.table is not None:
            try:
                self.d = int(self.table.schema.field("vector").type.list_size)
            except Exception:
                # Keep d unknown on reload if schema does not expose fixed vector size.
                self.d = None

    def search(self, query_vector, k):
        if self.table is None:
            self.load_index()

        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim == 2:
            if query_vector.shape[0] != 1:
                raise ValueError("Query embedding must be a 1D vector or shape (1, d)")
            query_vector = query_vector[0]
        if query_vector.ndim != 1:
            raise ValueError("Query embedding must be a 1D vector")
        if self.d is not None and query_vector.shape[0] != self.d:
            raise ValueError(
                "Query dimension mismatch: "
                f"expected {self.d}, got {query_vector.shape[0]}"
            )

        results = (
            self.table.search(query_vector.tolist(), vector_column_name="vector")
            .limit(int(k))
            .to_list()
        )
        distances = [float(r.get("_distance", 0.0)) for r in results]
        pdf_names = [r["pdf_name"] for r in results]
        pages = [int(r["page"]) for r in results]
        return distances, pdf_names, pages

    def total_entries(self):
        if self.table is None:
            self.load_index()
        return self.table.count_rows()

    def get_vectors_for_pdf_page_counts(self, pdf_page_counts):
        if self.table is None:
            self.load_index()
        if not pdf_page_counts:
            return {}

        candidates = {}
        rows = self.table.to_list()
        for row in rows:
            pdf_name = str(row.get("pdf_name", ""))
            if not pdf_name:
                continue
            max_pages = pdf_page_counts.get(pdf_name)
            if max_pages is None:
                continue
            try:
                page_num = int(row.get("page", -1))
            except (TypeError, ValueError):
                continue
            if page_num < 0 or page_num >= int(max_pages):
                continue

            vector = np.asarray(row.get("vector", []), dtype=np.float32)
            if pdf_name not in candidates:
                candidates[pdf_name] = []
            candidates[pdf_name].append((str(page_num), vector))
        return candidates


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
        # No JVM calls here — the gunicorn master may fork workers, and the
        # JVM's internal daemon threads (GC, finalizer) don't survive fork(),
        # leaving it in a broken state.  All JVM work is deferred to
        # _ensure_ready(), which runs lazily inside each worker process.
        self.index_keyword_directory = index_keyword_directory
        os.makedirs(self.index_keyword_directory, exist_ok=True)
        self._dir = None
        self._analyzer = None
        self._writer = None
        self._reader = None
        self._searcher = None
        # PID where the search index was opened; triggers re-init after fork.
        self._ready_pid = None
        # Ensures only one thread per process performs the full init.
        self._init_lock = threading.Lock()

    def _attach(self):
        """Start the JVM (if needed) and attach the current thread."""
        _load_lucene()
        env = lucene.getVMEnv()
        if env is not None:
            env.attachCurrentThread()

    def _ensure_ready(self):
        """
        Lazily start the JVM and open the search index, once per process.

        When gunicorn uses preload_app=True the Server object is created in the
        master process, which then fork()s workers. The JVM's internal daemon
        threads (GC, finalizer, etc.) are not copied by fork(), leaving it in
        an unreliable state where some JNI calls work and others deadlock.
        By deferring all JVM work to the first actual use inside each worker we
        guarantee that initVM() runs fresh in every process.
        """
        pid = os.getpid()
        if self._ready_pid == pid:
            self._attach()
            return

        with self._init_lock:
            if self._ready_pid == pid:
                self._attach()
                return

            self._attach()
            self._dir = FSDirectory.open(Paths.get(self.index_keyword_directory))
            self._analyzer = StandardAnalyzer()

            try:
                reader = DirectoryReader.open(self._dir)
            except Exception:
                # Index does not exist yet; create an empty one.
                cfg = IndexWriterConfig(self._analyzer)
                cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
                w = IndexWriter(self._dir, cfg)
                w.commit()
                w.close()
                reader = DirectoryReader.open(self._dir)

            searcher = IndexSearcher(reader)
            searcher.setSimilarity(BM25Similarity())
            self._reader = reader
            self._searcher = searcher
            self._ready_pid = pid

    def _open_dir(self):
        if self._dir is None:
            self._dir = FSDirectory.open(Paths.get(self.index_keyword_directory))

    def build_index(self):
        """Create/open the index for writing."""
        self._attach()
        self._open_dir()
        if self._analyzer is None:
            self._analyzer = StandardAnalyzer()

        if self._writer is None:
            cfg = IndexWriterConfig(self._analyzer)
            cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
            self._writer = IndexWriter(self._dir, cfg)

    def add_batch(self, texts, pdf_names, pages):
        """Adds documents to the Lucene index."""
        self._attach()
        if self._writer is None:
            self.build_index()

        for text, pdf_name, page in zip(texts, pdf_names, pages, strict=False):
            doc = Document()

            # No need to store the text since we only search for it, not return it.
            doc.add(TextField("text", text if text is not None else "", Field.Store.NO))
            doc.add(
                StringField(
                    "pdf_name",
                    pdf_name if pdf_name is not None else "",
                    Field.Store.YES,
                )
            )
            doc.add(StoredField("page", int(page)))
            self._writer.addDocument(doc)

    def save_index(self):
        """Commit changes and close writer."""
        self._attach()
        if self._writer is not None:
            self._writer.commit()
            self._writer.close()
            self._writer = None

    def load_index(self):
        # Deferred to _ensure_ready() on first search/total_entries call,
        # so the JVM is never started in the gunicorn master process.
        pass

    def search(self, query, k):
        """Search the index for the query string, returning up to k results."""
        self._ensure_ready()

        parser = QueryParser("text", self._analyzer)
        try:
            q = parser.parse(query)
        except Exception:
            q = parser.parse(QueryParser.escape(query))

        top_docs = self._searcher.search(q, int(k))
        hits = top_docs.scoreDocs

        scores, pdf_names, pages = [], [], []
        stored = self._searcher.storedFields()
        for sd in hits:
            doc = stored.document(sd.doc)
            scores.append(float(sd.score))
            pdf_names.append(doc.get("pdf_name"))
            pages.append(str(doc.get("page")))

        return scores, pdf_names, pages

    def total_entries(self):
        self._ensure_ready()
        return int(self._reader.numDocs())


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
    def search(self, pdf_names, predicates):
        """
        Return the metadata for the pdfs in 'pdf_names' that satisfy all 'predicates'.
        """

    @abstractmethod
    def total_entries(self):
        """
        Returns the total number of documents in the index.
        :return: Total number of embeddings.
        """

    @abstractmethod
    def count_filtered_documents(self, filter=None):
        """
        Return the total number of distinct pdf_name documents that satisfy
        the provided filter.
        """

    @abstractmethod
    def get_filtered_pdf_page_counts(self, filter=None):
        """
        Return {pdf_name: page_count} for all PDFs that satisfy the provided
        filter. If a PDF appears multiple times (multiple crawl dates), the
        maximum page_count is returned for that PDF.
        """

    @abstractmethod
    def upsert_vectors_batch(self, embedding_type, pdf_names, pages, vectors):
        """
        Persist vectors for (embedding_type, pdf_name, page) records.
        """

    @abstractmethod
    def get_vectors_for_pdf_page_counts(self, embedding_type, pdf_page_counts):
        """
        Return {pdf_name: [(page, embedding_vector), ...]} for pages where
        page < pdf_page_counts[pdf_name] and matching embedding_type.
        """
        
    @staticmethod
    def _normalize_crawl_date(date_str: str) -> str:
        """Truncate crawl_date to YYYYMMDD, stripping any trailing time component."""
        return date_str[:8]

def _normalize_filter_key(filter_dict):
    if not filter_dict:
        return ()
    pairs = []
    for key in sorted(filter_dict.keys()):
        value = filter_dict[key]
        if value is not None:
            pairs.append((key, str(value)))
    return tuple(pairs)

class SQLiteMetadataIndex(AbstractMetadataIndex):
    def __init__(self, index_metadata_directory):
        self.index_metadata_directory = index_metadata_directory
        self.db_path = os.path.join(self.index_metadata_directory, "metadata.db")
        self._metadata_table = "metadata"
        self._metadata_sub_domain_table = "metadata_sub_domain"
        self._metadata_crawl_date_table = "metadata_crawl_date"
        self._metadata_vectors_table = "metadata_vectors"
        self._idx_metadata_pdf_name = "idx_metadata_pdf_name"
        self._idx_sub_domain = "idx_sub_domain"
        self._idx_crawl_date = "idx_crawl_date"
        self._idx_vectors_embedding_pdf_page = "idx_vectors_embedding_pdf_page"
        self.conn = None
        self.cursor = None
        self._total_entries = -1
        self._filtered_document_count_cache = {}
        self._filtered_pdf_page_counts_cache = {}
        self._vector_cache = {}

    def _reset_filtered_caches(self):
        self._filtered_document_count_cache.clear()
        self._filtered_pdf_page_counts_cache.clear()

    @staticmethod
    def _date_bound(value, include_end_of_day=False):
        normalized = str(value).replace("-", "")
        if include_end_of_day:
            return normalized + "999999"
        return normalized

    def _apply_filter_clauses(
        self,
        query,
        params,
        filter_dict,
        *,
        sub_domain_col="sub_domain",
        date_col="crawl_date",
        include_end_of_day=False,
    ):
        if not filter_dict:
            return query, params

        for key, value in filter_dict.items():
            if value is None:
                continue
            if key == "sub_domain":
                query += f" AND {sub_domain_col}=?"
                params.append(value)
            elif key == "crawled_after":
                query += f" AND {date_col}>=?"
                params.append(self._date_bound(value))
            elif key == "crawled_before":
                query += f" AND {date_col}<=?"
                params.append(
                    self._date_bound(value, include_end_of_day=include_end_of_day)
                )
        return query, params

    def build_index(self):
        if not os.path.exists(self.index_metadata_directory):
            os.makedirs(self.index_metadata_directory)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode = OFF;")
        self.conn.execute("PRAGMA synchronous = OFF;")
        self.cursor = self.conn.cursor()
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._metadata_table} (
                crawl_url TEXT,
                crawl_date TEXT,
                pdf_name TEXT,
                sub_domain TEXT,
                page_count INTEGER,
                s3_url TEXT
            );
        """)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._metadata_sub_domain_table} (
                pdf_name TEXT,
                sub_domain TEXT,
                page_count INTEGER
            );
        """)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._metadata_crawl_date_table} (
                pdf_name TEXT,
                crawl_date TEXT,
                page_count INTEGER
            );
        """)
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._metadata_vectors_table} (
                embedding_type TEXT,
                pdf_name TEXT,
                page INTEGER,
                vector BLOB,
                PRIMARY KEY (embedding_type, pdf_name, page)
            );
        """)
        self.conn.commit()

    def add_batch(self, metadata_dicts):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        to_insert_metadata = [
            (
                md.get("crawl_url", ""),
                self._normalize_crawl_date(md.get("crawl_date", "")),
                md.get("pdf_name", ""),
                md.get("sub_domain", ""),
                md.get("page_count", 0),
                md.get("s3_url", ""),
            )
            for md in metadata_dicts
        ]
        to_insert_sub_domain = [
            (
                md.get("pdf_name", ""),
                md.get("sub_domain", ""),
                md.get("page_count", 0),
            )
            for md in metadata_dicts
        ]
        to_insert_crawl_date = [
            (
                md.get("pdf_name", ""),
                md.get("crawl_date", ""),
                md.get("page_count", 0),
            )
            for md in metadata_dicts
        ]

        self.cursor.executemany(
            f"INSERT INTO {self._metadata_table} ("
            "crawl_url, crawl_date, pdf_name, sub_domain, page_count, s3_url"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            to_insert_metadata,
        )
        self.cursor.executemany(
            f"INSERT INTO {self._metadata_sub_domain_table} ("
            "pdf_name, sub_domain, page_count"
            ") VALUES (?, ?, ?)",
            to_insert_sub_domain,
        )
        self.cursor.executemany(
            f"INSERT INTO {self._metadata_crawl_date_table} ("
            "pdf_name, crawl_date, page_count"
            ") VALUES (?, ?, ?)",
            to_insert_crawl_date,
        )
        self.conn.commit()
        self._reset_filtered_caches()

    def load_index(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        if self._total_entries == -1:
            self.cursor.execute(f"SELECT COUNT(*) FROM {self._metadata_table}")
            self._total_entries = self.cursor.fetchone()[0]

    def save_index(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._idx_metadata_pdf_name}
            ON {self._metadata_table} (pdf_name);""")
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._idx_sub_domain}
            ON {self._metadata_sub_domain_table} (sub_domain);""")
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._idx_crawl_date}
            ON {self._metadata_crawl_date_table} (crawl_date);""")
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._idx_vectors_embedding_pdf_page}
            ON {self._metadata_vectors_table} (embedding_type, pdf_name, page);""")

        self.cursor.execute("""PRAGMA journal_mode = WAL;""")
        self.cursor.execute("""PRAGMA optimize;""")

        self.conn.commit()

    def _metadata_table_for_filter(self, filter_dict):
        if not filter_dict:
            return self._metadata_table

        has_sub_domain = filter_dict.get("sub_domain") is not None
        has_crawl_date = (
            filter_dict.get("crawled_after") is not None
            or filter_dict.get("crawled_before") is not None
        )

        if has_sub_domain and not has_crawl_date:
            return self._metadata_sub_domain_table
        if has_crawl_date and not has_sub_domain:
            return self._metadata_crawl_date_table
        return self._metadata_table

    def search(self, pdf_names, filter=None):
        if len(pdf_names) == 0:
            return {}
    def search(self, pdf_names, predicates: list[Predicate] | None = None):
        cursor = self.conn.cursor()
        placeholders = ",".join(["?"] * len(pdf_names))
        query = (
            "SELECT crawl_url, crawl_date, pdf_name, sub_domain, s3_url, page_count "
            f"FROM {self._metadata_table} WHERE pdf_name IN ({placeholders})"
        )
        params = list(pdf_names)
        query, params = self._apply_filter_clauses(
            query,
            params,
            filter,
            date_col="crawl_date",
            include_end_of_day=True,
        )
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        metadata = {}
        for row in rows:
            pdf_name = row[2]
            row_dict = {
                "crawl_url": row[0],
                "crawl_date": row[1],
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

    def count_filtered_documents(self, filter=None):
        key = _normalize_filter_key(filter)
        if key in self._filtered_document_count_cache:
            return self._filtered_document_count_cache[key]

        has_sub_domain = filter and filter.get("sub_domain") is not None
        has_crawl_date = filter and (
            filter.get("crawled_after") is not None
            or filter.get("crawled_before") is not None
        )

        if has_sub_domain and has_crawl_date:
            total_docs = self.count_filtered_documents(None)
            if total_docs <= 0:
                self._filtered_document_count_cache[key] = 0
                return 0

            sub_docs = self.count_filtered_documents(
                {"sub_domain": filter["sub_domain"]}
            )
            date_only_filter = {}
            if filter.get("crawled_after") is not None:
                date_only_filter["crawled_after"] = filter["crawled_after"]
            if filter.get("crawled_before") is not None:
                date_only_filter["crawled_before"] = filter["crawled_before"]
            date_docs = self.count_filtered_documents(date_only_filter)

            result = int(round((sub_docs * date_docs) / max(total_docs, 1)))
            result = max(0, min(result, sub_docs, date_docs, total_docs))
            self._filtered_document_count_cache[key] = result
            return result

        metadata_table = self._metadata_table_for_filter(filter)
        query = f"SELECT COUNT(DISTINCT pdf_name) FROM {metadata_table} WHERE 1=1"
        params = []
        query, params = self._apply_filter_clauses(
            query,
            params,
            filter,
            date_col="crawl_date",
            include_end_of_day=True,
        )

        self.cursor.execute(query, params)
        value = self.cursor.fetchone()
        result = int(value[0]) if value and value[0] is not None else 0
        self._filtered_document_count_cache[key] = result
        return result

    def get_filtered_pdf_page_counts(self, filter=None):
        key = _normalize_filter_key(filter)
        if key in self._filtered_pdf_page_counts_cache:
            return self._filtered_pdf_page_counts_cache[key]

        has_sub_domain = filter and filter.get("sub_domain") is not None
        has_crawl_date = filter and (
            filter.get("crawled_after") is not None
            or filter.get("crawled_before") is not None
        )

        if has_sub_domain and has_crawl_date:
            query = (
                "SELECT sd.pdf_name, MAX(sd.page_count) "
                f"FROM {self._metadata_sub_domain_table} sd "
                f"INNER JOIN {self._metadata_crawl_date_table} cd "
                "ON sd.pdf_name = cd.pdf_name "
                "WHERE 1=1"
            )
            params = []
            query += " AND sd.sub_domain=?"
            params.append(filter["sub_domain"])
            if filter.get("crawled_after") is not None:
                query += " AND cd.crawl_date>=?"
                params.append(self._date_bound(filter["crawled_after"]))
            if filter.get("crawled_before") is not None:
                query += " AND cd.crawl_date<=?"
                params.append(
                    self._date_bound(filter["crawled_before"], include_end_of_day=True)
                )
            query += " GROUP BY sd.pdf_name"

            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            result = {str(row[0]): int(row[1]) for row in rows}
            self._filtered_pdf_page_counts_cache[key] = result
            return result

        metadata_table = self._metadata_table_for_filter(filter)
        query = f"SELECT pdf_name, MAX(page_count) FROM {metadata_table} WHERE 1=1"
        params = []
        query, params = self._apply_filter_clauses(
            query,
            params,
            filter,
            date_col="crawl_date",
            include_end_of_day=True,
        )
        query += " GROUP BY pdf_name"

        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()
        result = {str(row[0]): int(row[1]) for row in rows}
        self._filtered_pdf_page_counts_cache[key] = result
        return result

    def total_entries(self):
        if self._total_entries == -1:
            self.load_index()
        return self._total_entries

    def upsert_vectors_batch(self, embedding_type, pdf_names, pages, vectors):
        if not pdf_names:
            return
        rows = []
        for pdf_name, page, vector in zip(pdf_names, pages, vectors, strict=False):
            vector_np = np.asarray(vector, dtype=np.float32)
            if vector_np.ndim > 1:
                vector_np = vector_np.reshape(-1)
            rows.append(
                (
                    str(embedding_type),
                    str(pdf_name),
                    int(page),
                    vector_np.tobytes(),
                )
            )

        self.cursor.executemany(
            f"""
            INSERT OR REPLACE INTO {self._metadata_vectors_table}
                (embedding_type, pdf_name, page, vector)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        self._vector_cache.clear()

    def get_vectors_for_pdf_page_counts(self, embedding_type, pdf_page_counts):
        if not pdf_page_counts:
            return {}

        cache_key = (str(embedding_type), tuple(sorted(pdf_page_counts.items())))
        if cache_key in self._vector_cache:
            return self._vector_cache[cache_key]

        pdf_names = list(pdf_page_counts.keys())
        placeholders = ",".join(["?"] * len(pdf_names))
        query = (
            "SELECT pdf_name, page, vector "
            f"FROM {self._metadata_vectors_table} "
            "WHERE embedding_type=? "
            f"AND pdf_name IN ({placeholders})"
        )
        params = [str(embedding_type)] + pdf_names
        self.cursor.execute(query, params)
        rows = self.cursor.fetchall()

        result = {}
        for pdf_name, page, vector_blob in rows:
            max_pages = pdf_page_counts.get(pdf_name)
            if max_pages is None or int(page) >= int(max_pages):
                continue
            vector = np.frombuffer(vector_blob, dtype=np.float32).copy()
            if pdf_name not in result:
                result[pdf_name] = []
            result[pdf_name].append((str(page), vector))

        self._vector_cache[cache_key] = result
        return result


class DuckDBMetadataIndex(AbstractMetadataIndex):
    def __init__(self, index_metadata_directory):
        self.index_metadata_directory = index_metadata_directory
        self.db_path = os.path.join(self.index_metadata_directory, "metadata.duckdb")
        self._metadata_table = "metadata"
        self._metadata_sub_domain_table = "metadata_sub_domain"
        self._metadata_crawl_date_table = "metadata_crawl_date"
        self._metadata_vectors_table = "metadata_vectors"
        self._idx_metadata_pdf_name = "idx_metadata_pdf_name"
        self._idx_sub_domain = "idx_sub_domain"
        self._idx_crawl_date = "idx_crawl_date"
        self._idx_vectors_embedding_pdf_page = "idx_vectors_embedding_pdf_page"
        self.conn = None
        self._total_entries = -1
        self._filtered_document_count_cache = {}
        self._filtered_pdf_page_counts_cache = {}
        self._vector_cache = {}

    def _reset_filtered_caches(self):
        self._filtered_document_count_cache.clear()
        self._filtered_pdf_page_counts_cache.clear()

    @staticmethod
    def _date_bound(value, include_end_of_day=False):
        normalized = str(value).replace("-", "")
        if include_end_of_day:
            return normalized + "999999"
        return normalized

    def _apply_filter_clauses(
        self,
        query,
        params,
        filter_dict,
        *,
        sub_domain_col="sub_domain",
        date_col="crawl_date",
        include_end_of_day=False,
    ):
        if not filter_dict:
            return query, params

        for key, value in filter_dict.items():
            if value is None:
                continue
            if key == "sub_domain":
                query += f" AND {sub_domain_col} = ?"
                params.append(value)
            elif key == "crawled_after":
                query += f" AND {date_col} >= ?"
                params.append(self._date_bound(value))
            elif key == "crawled_before":
                query += f" AND {date_col} <= ?"
                params.append(
                    self._date_bound(value, include_end_of_day=include_end_of_day)
                )
        return query, params

    def _connect(self):
        if self.conn is None:
            os.makedirs(self.index_metadata_directory, exist_ok=True)
            self.conn = duckdb.connect(self.db_path)

    def build_index(self):
        self._connect()
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._metadata_table} (
                crawl_url TEXT,
                crawl_date TEXT,
                pdf_name TEXT,
                sub_domain TEXT,
                page_count INTEGER,
                s3_url TEXT
            );
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._metadata_sub_domain_table} (
                pdf_name TEXT,
                sub_domain TEXT,
                page_count INTEGER
            );
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._metadata_crawl_date_table} (
                pdf_name TEXT,
                crawl_date TEXT,
                page_count INTEGER
            );
        """)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._metadata_vectors_table} (
                embedding_type TEXT,
                pdf_name TEXT,
                page INTEGER,
                vector BLOB
            );
        """)

    def add_batch(self, metadata_dicts):
        self._connect()
        # DuckDB is extremely slow to insert directly. (intractably slow)
        # Inserting via a PyArrow table is orders of magnitude faster.
        arrow_table = pa.table(
            {
                "crawl_url": [md.get("crawl_url", "") for md in metadata_dicts],
                "crawl_date": [
                    self._normalize_crawl_date(md.get("crawl_date", ""))
                    for md in metadata_dicts
                ],
                "pdf_name": [md.get("pdf_name", "") for md in metadata_dicts],
                "sub_domain": [md.get("sub_domain", "") for md in metadata_dicts],
                "page_count": pa.array(
                    [md.get("page_count", 0) for md in metadata_dicts], type=pa.int32()
                ),
                "s3_url": [md.get("s3_url", "") for md in metadata_dicts],
            }
        )
        self.conn.register("_batch", arrow_table)
        self.conn.execute(f"INSERT INTO {self._metadata_table} SELECT * FROM _batch")
        self.conn.execute(f"""
            INSERT INTO {self._metadata_sub_domain_table}
            SELECT pdf_name, sub_domain, page_count FROM _batch
        """)
        self.conn.execute(f"""
            INSERT INTO {self._metadata_crawl_date_table}
            SELECT pdf_name, crawl_date, page_count FROM _batch
        """)
        self.conn.unregister("_batch")
        self._reset_filtered_caches()

    def load_index(self):
        self._connect()
        if self._total_entries == -1:
            result = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._metadata_table}"
            ).fetchone()
            self._total_entries = result[0] if result else 0

    def _metadata_table_for_filter(self, filter_dict):
        if not filter_dict:
            return self._metadata_table

        has_sub_domain = filter_dict.get("sub_domain") is not None
        has_crawl_date = (
            filter_dict.get("crawled_after") is not None
            or filter_dict.get("crawled_before") is not None
        )

        if has_sub_domain and not has_crawl_date:
            return self._metadata_sub_domain_table
        if has_crawl_date and not has_sub_domain:
            return self._metadata_crawl_date_table
        return self._metadata_table

    def save_index(self):
        self._connect()
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._idx_metadata_pdf_name}
            ON {self._metadata_table} (pdf_name);
                            """)
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._idx_sub_domain}
            ON {self._metadata_sub_domain_table} (sub_domain);
                            """)
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._idx_crawl_date}
            ON {self._metadata_crawl_date_table} (crawl_date);
                            """)
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {self._idx_vectors_embedding_pdf_page}
            ON {self._metadata_vectors_table} (embedding_type, pdf_name, page);
                            """)
        self.conn.checkpoint()

    def search(self, pdf_names: list[str], predicates: list[Predicate] | None = None):
        self._connect()
        if len(pdf_names) == 0:
            return {}
        placeholders = ", ".join(["?"] * len(pdf_names))
        query = (
            "SELECT crawl_url, crawl_date, pdf_name, sub_domain, s3_url, page_count "
            f"FROM {self._metadata_table} WHERE pdf_name IN ({placeholders})"
        )
        params = list(pdf_names)
        query, params = self._apply_filter_clauses(
            query,
            params,
            filter,
            date_col="crawl_date",
            include_end_of_day=True,
        )
        rows = self.conn.execute(query, params).fetchall()
        metadata = {}
        for row in rows:
            pdf_name = row[2]
            row_dict = {
                "crawl_url": row[0],
                "crawl_date": row[1],
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

    def count_filtered_documents(self, filter=None):
        self._connect()
        key = _normalize_filter_key(filter)
        if key in self._filtered_document_count_cache:
            return self._filtered_document_count_cache[key]

        has_sub_domain = filter and filter.get("sub_domain") is not None
        has_crawl_date = filter and (
            filter.get("crawled_after") is not None
            or filter.get("crawled_before") is not None
        )

        if has_sub_domain and has_crawl_date:
            total_docs = self.count_filtered_documents(None)
            if total_docs <= 0:
                self._filtered_document_count_cache[key] = 0
                return 0

            sub_docs = self.count_filtered_documents(
                {"sub_domain": filter["sub_domain"]}
            )
            date_only_filter = {}
            if filter.get("crawled_after") is not None:
                date_only_filter["crawled_after"] = filter["crawled_after"]
            if filter.get("crawled_before") is not None:
                date_only_filter["crawled_before"] = filter["crawled_before"]
            date_docs = self.count_filtered_documents(date_only_filter)

            result = int(round((sub_docs * date_docs) / max(total_docs, 1)))
            result = max(0, min(result, sub_docs, date_docs, total_docs))
            self._filtered_document_count_cache[key] = result
            return result

        metadata_table = self._metadata_table_for_filter(filter)
        query = f"SELECT COUNT(DISTINCT pdf_name) FROM {metadata_table} WHERE 1=1"
        params = []
        query, params = self._apply_filter_clauses(
            query,
            params,
            filter,
            date_col="crawl_date",
            include_end_of_day=True,
        )

        row = self.conn.execute(query, params).fetchone()
        result = int(row[0]) if row and row[0] is not None else 0
        self._filtered_document_count_cache[key] = result
        return result

    def get_filtered_pdf_page_counts(self, filter=None):
        self._connect()
        key = _normalize_filter_key(filter)
        if key in self._filtered_pdf_page_counts_cache:
            return self._filtered_pdf_page_counts_cache[key]

        has_sub_domain = filter and filter.get("sub_domain") is not None
        has_crawl_date = filter and (
            filter.get("crawled_after") is not None
            or filter.get("crawled_before") is not None
        )

        if has_sub_domain and has_crawl_date:
            query = (
                "SELECT sd.pdf_name, MAX(sd.page_count) "
                f"FROM {self._metadata_sub_domain_table} sd "
                f"INNER JOIN {self._metadata_crawl_date_table} cd "
                "ON sd.pdf_name = cd.pdf_name "
                "WHERE 1=1"
            )
            params = []
            query += " AND sd.sub_domain = ?"
            params.append(filter["sub_domain"])
            if filter.get("crawled_after") is not None:
                query += " AND cd.crawl_date >= ?"
                params.append(self._date_bound(filter["crawled_after"]))
            if filter.get("crawled_before") is not None:
                query += " AND cd.crawl_date <= ?"
                params.append(
                    self._date_bound(filter["crawled_before"], include_end_of_day=True)
                )
            query += " GROUP BY sd.pdf_name"

            rows = self.conn.execute(query, params).fetchall()
            result = {str(row[0]): int(row[1]) for row in rows}
            self._filtered_pdf_page_counts_cache[key] = result
            return result

        metadata_table = self._metadata_table_for_filter(filter)
        query = f"SELECT pdf_name, MAX(page_count) FROM {metadata_table} WHERE 1=1"
        params = []
        query, params = self._apply_filter_clauses(
            query,
            params,
            filter,
            date_col="crawl_date",
            include_end_of_day=True,
        )
        query += " GROUP BY pdf_name"

        rows = self.conn.execute(query, params).fetchall()
        result = {str(row[0]): int(row[1]) for row in rows}
        self._filtered_pdf_page_counts_cache[key] = result
        return result

    def total_entries(self):
        if self._total_entries == -1:
            self.load_index()
        return self._total_entries

    def upsert_vectors_batch(self, embedding_type, pdf_names, pages, vectors):
        if not pdf_names:
            return
        self._connect()

        embedding_types = []
        out_pdf_names = []
        out_pages = []
        out_vectors = []
        for pdf_name, page, vector in zip(pdf_names, pages, vectors, strict=False):
            vector_np = np.asarray(vector, dtype=np.float32)
            if vector_np.ndim > 1:
                vector_np = vector_np.reshape(-1)
            embedding_types.append(str(embedding_type))
            out_pdf_names.append(str(pdf_name))
            out_pages.append(int(page))
            out_vectors.append(vector_np.tobytes())

        arrow_table = pa.table(
            {
                "embedding_type": embedding_types,
                "pdf_name": out_pdf_names,
                "page": out_pages,
                "vector": out_vectors,
            }
        )
        self.conn.register("_vector_batch", arrow_table)
        self.conn.execute(
            f"""
            DELETE FROM {self._metadata_vectors_table}
            USING _vector_batch
            WHERE {self._metadata_vectors_table}.embedding_type =
                  _vector_batch.embedding_type
              AND {self._metadata_vectors_table}.pdf_name =
                  _vector_batch.pdf_name
              AND {self._metadata_vectors_table}.page =
                  _vector_batch.page
            """
        )
        self.conn.execute(
            f"INSERT INTO {self._metadata_vectors_table} SELECT * FROM _vector_batch"
        )
        self.conn.unregister("_vector_batch")
        self._vector_cache.clear()

    def get_vectors_for_pdf_page_counts(self, embedding_type, pdf_page_counts):
        if not pdf_page_counts:
            return {}
        self._connect()

        cache_key = (str(embedding_type), tuple(sorted(pdf_page_counts.items())))
        if cache_key in self._vector_cache:
            return self._vector_cache[cache_key]

        pdf_names = list(pdf_page_counts.keys())
        placeholders = ", ".join(["?"] * len(pdf_names))
        query = (
            f"SELECT pdf_name, page, vector FROM {self._metadata_vectors_table} "
            "WHERE embedding_type = ? "
            f"AND pdf_name IN ({placeholders})"
        )
        params = [str(embedding_type)] + pdf_names
        rows = self.conn.execute(query, params).fetchall()

        result = {}
        for pdf_name, page, vector_blob in rows:
            max_pages = pdf_page_counts.get(pdf_name)
            if max_pages is None or int(page) >= int(max_pages):
                continue
            vector = np.frombuffer(vector_blob, dtype=np.float32).copy()
            if pdf_name not in result:
                result[pdf_name] = []
            result[pdf_name].append((str(page), vector))

        self._vector_cache[cache_key] = result
        return result
