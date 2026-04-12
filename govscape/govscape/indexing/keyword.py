import os
import sqlite3
import threading
from abc import ABC, abstractmethod

import pyarrow as pa
from lancedb import connect
from lancedb.query import MatchQuery, PhraseQuery
from whoosh.fields import ID, NUMERIC, TEXT, Schema
from whoosh.filedb.filestore import FileStorage
from whoosh.index import create_in
from whoosh.qparser import QueryParser

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
                LuceneQueryParser, \
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
                QueryParser as LuceneQueryParser,
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

        parser = LuceneQueryParser("text", self._analyzer)
        try:
            q = parser.parse(query)
        except Exception:
            q = parser.parse(LuceneQueryParser.escape(query))

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
