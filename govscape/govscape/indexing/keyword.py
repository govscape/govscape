# AI modified: 2026-04-26 00:00:00 341724af
# AI modified: 2026-04-26T22:00:43Z eac4f332
import contextlib
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
from whoosh.query import And, Or, Term

lucene = None
LuceneTerm = None
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
                BooleanClause, \
                BooleanQuery, \
                TermQuery, \
                LuceneQueryParser, \
                BM25Similarity
            global LuceneTerm

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
            from org.apache.lucene.index import (  # type: ignore[import]
                Term as LuceneTerm,
            )
            from org.apache.lucene.queryparser.classic import (  # type: ignore[import]
                QueryParser as LuceneQueryParser,
            )
            from org.apache.lucene.search import (  # type: ignore[import]
                BooleanClause,
                BooleanQuery,
                IndexSearcher,
                TermQuery,
            )
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
    def add_batch(self, texts, digests, pages):
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
        :return: A tuple of distances, digests, and pages.
        """

    @abstractmethod
    def search_filtered(self, query_vector, k, allowed_names):
        """Search for up to k results while restricting output to allowed_names."""

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
                pa.field("name", pa.string()),
                pa.field("page", pa.int32()),
            ]
        )
        self.table = self.db.create_table(self.table_name, schema=schema)
        self.table.create_fts_index("text", with_position=True)

    def add_batch(self, texts, digests, pages):
        if self.table is None:
            self.build_index()
        table_fields = set(self.table.schema.names)
        rows = [
            {
                "text": text,
                "pdf_name": digest,
                "page": int(page),
                **({"name": digest} if "name" in table_fields else {}),
            }
            for text, digest, page in zip(texts, digests, pages, strict=False)
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

    def _run_search(self, query, k, allowed_names=None):
        if self.table is None:
            self.load_index()
        if not query:
            return [], [], []

        query_obj = None
        if query[0] == '"' and query[-1] == '"':
            query_obj = PhraseQuery(query, "text")
        else:
            query_obj = MatchQuery(query, "text")

        result_query = self.table.search(
            query_obj,
            fts_columns="text",
            query_type="fts",
            vector_column_name="",
        )

        if allowed_names is not None:
            names = sorted(str(name) for name in allowed_names)
            if not names:
                return [], [], []
            field_name = "name" if "name" in self.table.schema.names else "pdf_name"
            escaped = ["'" + name.replace("'", "''") + "'" for name in names]
            where_clause = f"{field_name} IN ({', '.join(escaped)})"
            with contextlib.suppress(Exception):
                result_query = result_query.where(where_clause)

        results = result_query.limit(k).to_list()
        if allowed_names is not None:
            allowed_set = {str(name) for name in allowed_names}
            results = [r for r in results if r.get("pdf_name") in allowed_set]

        scores = [r.get("_score", 0.0) for r in results]
        digests = [r["pdf_name"] for r in results]
        pages = [str(r["page"]) for r in results]
        return scores, digests, pages

    def search(self, query, k):
        return self._run_search(query, k, allowed_names=None)

    def search_filtered(self, query, k, allowed_names):
        return self._run_search(query, k, allowed_names=allowed_names)

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
                page_count,
                name UNINDEXED
            );
            """
        )
        self.conn.commit()

    def _has_name_column(self):
        if self.conn is None:
            self.load_index()
        rows = self.cursor.execute("PRAGMA table_info(fts_txt)").fetchall()
        return any(str(row[1]) == "name" for row in rows)

    def add_batch(self, texts, digests, pages):
        if self.conn is None:
            self.load_index()
        self.cursor.execute("BEGIN TRANSACTION;")
        has_name_column = self._has_name_column()
        for text, digest, page in zip(texts, digests, pages, strict=False):
            if has_name_column:
                self.cursor.execute(
                    "INSERT INTO fts_txt (text, pdf_name, page_count, name) "
                    "VALUES (?, ?, ?, ?)",
                    [text, digest, page, digest],
                )
            else:
                self.cursor.execute(
                    "INSERT INTO fts_txt (text, pdf_name, page_count) VALUES (?, ?, ?)",
                    [text, digest, page],
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
            row = self.cursor.fetchone()
            self._total_entries = row[0] if row and row[0] is not None else 0

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

    def _search_impl(self, query, k, allowed_names=None):
        query = self._clean_query(query)
        if not self.cursor:
            self.load_index()

        params = [query]
        name_column = "name" if self._has_name_column() else "pdf_name"
        filtered_clause = ""
        if allowed_names is not None:
            names = [str(name) for name in allowed_names]
            if not names:
                return [], [], []
            placeholders = ",".join(["?"] * len(names))
            filtered_clause = f" AND {name_column} IN ({placeholders})"
            params.extend(names)

        search_query = (
            "SELECT *, rank FROM fts_txt "
            "WHERE fts_txt MATCH ?"
            f"{filtered_clause} "
            "ORDER BY rank LIMIT ?"
        )
        params.append(k)

        try:
            rows = self.cursor.execute(search_query, params).fetchall()
        except sqlite3.ProgrammingError:
            self.load_index()
            rows = self.cursor.execute(search_query, params).fetchall()
        distances = []
        digests = []
        pages = []
        for row in rows:
            digests.append(row[1])
            pages.append(str(row[2]))
            distances.append(row[3])
        return distances, digests, pages

    def search(self, query, k):
        return self._search_impl(query, k, allowed_names=None)

    def search_filtered(self, query, k, allowed_names):
        return self._search_impl(query, k, allowed_names=allowed_names)

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
            text=TEXT(stored=True),
            pdf_name=ID(stored=True),
            name=ID(stored=True),
            page=NUMERIC(stored=True),
        )
        if not os.path.exists(self.index_keyword_directory):
            os.makedirs(self.index_keyword_directory)
        self.index = create_in(self.index_keyword_directory, schema)

    def add_batch(self, texts, digests, pages):
        if self.index is None:
            self.build_index()
        # Whoosh's multi-process writer is prone to finish_subsegment races when the
        # caller exits or crashes early, so stick with a single-process writer for
        # stability.
        writer = self.index.writer(procs=12, limitmb=512)
        schema_names = set(self.index.schema.names())
        for text, digest, page in zip(texts, digests, pages, strict=False):
            doc = {"text": text, "pdf_name": digest, "page": page}
            if "name" in schema_names:
                doc["name"] = digest
            writer.add_document(**doc)
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

    def _search_with_allowed_names(self, query, k, allowed_names=None):
        if self.index is None:
            self.load_index()
        with self.index.searcher() as searcher:
            parser = QueryParser("text", self.index.schema)
            q = parser.parse(query)
            if allowed_names is not None:
                names = [str(name) for name in allowed_names]
                if not names:
                    return [], [], []
                field_name = (
                    "name" if "name" in self.index.schema.names() else "pdf_name"
                )
                max_terms_per_query = 500
                all_rows = []
                for i in range(0, len(names), max_terms_per_query):
                    chunk = names[i : i + max_terms_per_query]
                    name_query = Or([Term(field_name, name) for name in chunk])
                    chunk_query = And([q, name_query])
                    chunk_results = searcher.search(chunk_query, limit=k)
                    all_rows.extend(
                        [
                            (row.score, row["pdf_name"], str(row["page"]))
                            for row in chunk_results
                        ]
                    )

                if not all_rows:
                    return [], [], []

                deduped_rows = {}
                for score, digest, page in all_rows:
                    row_key = (digest, page)
                    current = deduped_rows.get(row_key)
                    if current is None or score > current[0]:
                        deduped_rows[row_key] = (float(score), digest, page)

                ranked_rows = sorted(
                    deduped_rows.values(),
                    key=lambda row: row[0],
                    reverse=True,
                )[: int(k)]
                scores = [row[0] for row in ranked_rows]
                digests = [row[1] for row in ranked_rows]
                pages = [row[2] for row in ranked_rows]
                return scores, digests, pages

            results = searcher.search(q, limit=k)
            digests = [r["pdf_name"] for r in results]
            pages = [str(r["page"]) for r in results]
            scores = [r.score for r in results]
        return scores, digests, pages

    def search(self, query, k):
        return self._search_with_allowed_names(query, k, allowed_names=None)

    def search_filtered(self, query, k, allowed_names):
        return self._search_with_allowed_names(query, k, allowed_names=allowed_names)

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

    def add_batch(self, texts, digests, pages):
        """Adds documents to the Lucene index."""
        self._attach()
        if self._writer is None:
            self.build_index()

        for text, digest, page in zip(texts, digests, pages, strict=False):
            doc = Document()

            # No need to store the text since we only search for it, not return it.
            doc.add(TextField("text", text if text is not None else "", Field.Store.NO))
            doc.add(
                StringField(
                    "pdf_name",
                    digest if digest is not None else "",
                    Field.Store.YES,
                )
            )
            doc.add(
                StringField(
                    "name",
                    digest if digest is not None else "",
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

    def _execute_query(self, lucene_query, k):
        top_docs = self._searcher.search(lucene_query, int(k))
        hits = top_docs.scoreDocs

        scores, digests, pages = [], [], []
        stored = self._searcher.storedFields()
        for sd in hits:
            doc = stored.document(sd.doc)
            scores.append(float(sd.score))
            digests.append(doc.get("pdf_name"))
            pages.append(str(doc.get("page")))
        return scores, digests, pages

    def _parse_query(self, query):
        self._ensure_ready()

        parser = LuceneQueryParser("text", self._analyzer)
        try:
            return parser.parse(query)
        except Exception:
            return parser.parse(LuceneQueryParser.escape(query))

    def search(self, query, k):
        """Search the index for the query string, returning up to k results."""
        q = self._parse_query(query)
        return self._execute_query(q, k)

    def search_filtered(self, query, k, allowed_names):
        self._ensure_ready()
        names = [str(name) for name in allowed_names]
        if not names:
            return [], [], []

        base_query = self._parse_query(query)
        max_terms_per_query = 400
        score_rows = []
        for i in range(0, len(names), max_terms_per_query):
            chunk = names[i : i + max_terms_per_query]
            name_filter = BooleanQuery.Builder()
            for name in chunk:
                name_filter.add(
                    TermQuery(LuceneTerm("name", name)),
                    BooleanClause.Occur.SHOULD,
                )
                name_filter.add(
                    TermQuery(LuceneTerm("pdf_name", name)),
                    BooleanClause.Occur.SHOULD,
                )

            filtered_query = BooleanQuery.Builder()
            filtered_query.add(base_query, BooleanClause.Occur.MUST)
            filtered_query.add(name_filter.build(), BooleanClause.Occur.FILTER)
            scores, digests, pages = self._execute_query(filtered_query.build(), k)
            score_rows.extend(zip(scores, digests, pages, strict=False))

        if not score_rows:
            return [], [], []

        # keep best-scoring hit per (pdf,page), then return global top-k
        best_rows = {}
        for score, digest, page in score_rows:
            row_key = (digest, page)
            current = best_rows.get(row_key)
            if current is None or score > current[0]:
                best_rows[row_key] = (float(score), digest, str(page))

        ranked = sorted(best_rows.values(), key=lambda row: row[0], reverse=True)
        ranked = ranked[: int(k)]
        scores = [row[0] for row in ranked]
        digests = [row[1] for row in ranked]
        pages = [row[2] for row in ranked]
        return scores, digests, pages

    def total_entries(self):
        self._ensure_ready()
        return int(self._reader.numDocs())
