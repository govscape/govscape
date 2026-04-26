# AI modified: 2026-04-19 21:12:31 c1b6021e
# AI modified: 2026-04-20 00:00:00 c1b6021e
# AI modified: 2026-04-20 00:00:00 4a7a8111
"""Metadata index implementations used by serving and filtering planners."""

import os
import sqlite3
from abc import ABC, abstractmethod

import numpy as np

import duckdb
import pyarrow as pa

from ..query import EqualityPredicate, Predicate, RangePredicate
from .filter_specs import get_default_filter_specs


class AbstractMetadataIndex(ABC):
    @abstractmethod
    def __init__(self, index_metadata_directory):
        self.index_metadata_directory = index_metadata_directory

    @abstractmethod
    def build_index(self):
        """
        Instantiate the index in the directory 'self.index_metadata_directory'
        which should store the following data:
        url, crawl_date, digest, sub_domain
        """

    @abstractmethod
    def add_batch(self, metadata_dicts):
        """
        Add a batch of metadata dictionaries to the index which each
        contain url, crawl_date, digest, and sub_domain.
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
    def search(self, digests, predicates):
        """
        Return the metadata for the pdfs in 'digests' that satisfy all 'predicates'.
        """

    @abstractmethod
    def total_entries(self):
        """
        Returns the total number of documents in the index.
        :return: Total number of embeddings.
        """

    @abstractmethod
    def estimate_selectivity(self, predicates: list[Predicate] | None = None) -> float:
        """Estimate conjunctive predicate selectivity in [0, 1]."""

    @abstractmethod
    def get_candidate_pdf_names(
        self, predicates: list[Predicate] | None = None
    ) -> set[str]:
        """Return distinct pdf names that satisfy all predicates."""

    def upsert_vectors(self, vector_store_key, vectors, pdf_names, pages):
        """Upsert vector rows keyed by (vector_store_key, pdf_name, page)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support vector storage"
        )

    def get_vectors_for_pdf_names(self, vector_store_key, candidate_pdf_names):
        """Return (vectors, names, pages) for candidate PDF names."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support vector retrieval"
        )

    def vector_count(self, vector_store_key: str) -> int:
        """Return number of stored vectors for a vector_store_key."""
        return 0

    @staticmethod
    def _normalize_crawl_date(date_str: str) -> str:
        """Truncate crawl_date to YYYYMMDD, stripping any trailing time component."""
        return date_str.replace("-", "")[:8]


class SQLiteMetadataIndex(AbstractMetadataIndex):
    def __init__(self, index_metadata_directory, filter_table_specs=None):
        self.index_metadata_directory = index_metadata_directory
        self.db_path = os.path.join(self.index_metadata_directory, "metadata.db")
        self.conn = None
        self.cursor = None
        self._total_entries = -1
        self._total_documents = -1
        self.filter_table_specs = (
            filter_table_specs
            if filter_table_specs is not None
            else get_default_filter_specs()
        )

    def _normalize_value(self, field_name, value):
        if value is None:
            return None
        if field_name == "crawl_date":
            return self._normalize_crawl_date(str(value))
        return value

    def _predicate_sql(self, predicate):
        spec = self.filter_table_specs.get(predicate.field_name)
        if spec is None:
            raise ValueError(
                f"Unsupported metadata predicate field: {predicate.field_name}"
            )

        clauses = []
        params = []
        if isinstance(predicate, EqualityPredicate):
            if not spec.supports_exact:
                raise ValueError(
                    f"Field '{predicate.field_name}' does not support exact predicates"
                )
            clauses.append(f"{spec.value_column} = ?")
            params.append(self._normalize_value(predicate.field_name, predicate.value))
        elif isinstance(predicate, RangePredicate):
            if not spec.supports_range:
                raise ValueError(
                    f"Field '{predicate.field_name}' does not support range predicates"
                )
            if predicate.min_val is not None:
                clauses.append(f"{spec.value_column} >= ?")
                params.append(
                    self._normalize_value(predicate.field_name, predicate.min_val)
                )
            if predicate.max_val is not None:
                clauses.append(f"{spec.value_column} <= ?")
                params.append(
                    self._normalize_value(predicate.field_name, predicate.max_val)
                )
        else:
            raise TypeError(f"Unsupported predicate type: {type(predicate)}")

        return spec, clauses, params

    def _query_pdf_names_for_predicate(self, predicate):
        spec, clauses, params = self._predicate_sql(predicate)
        query = f"SELECT DISTINCT pdf_name FROM {spec.table_name}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return {row[0] for row in rows}

    def _count_pdf_names_for_predicate(self, predicate):
        spec, clauses, params = self._predicate_sql(predicate)
        query = f"SELECT COUNT(DISTINCT pdf_name) FROM {spec.table_name}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        return row[0] if row else 0

    def _total_documents_count(self):
        if self._total_documents != -1:
            return self._total_documents
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT pdf_name) FROM metadata")
        row = cursor.fetchone()
        self._total_documents = row[0] if row else 0
        return self._total_documents

    @staticmethod
    def _serialize_vector(vector):
        arr = np.asarray(vector, dtype=np.float32)
        return arr.tobytes(), int(arr.shape[0])

    @staticmethod
    def _deserialize_vector(vector_blob, vector_dim):
        return np.frombuffer(vector_blob, dtype=np.float32, count=vector_dim)

    def build_index(self):
        if not os.path.exists(self.index_metadata_directory):
            os.makedirs(self.index_metadata_directory)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode = OFF;")
        self.conn.execute("PRAGMA synchronous = OFF;")
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                crawl_url TEXT,
                crawl_date TEXT,
                digest TEXT,
                pretty_name TEXT,
                sub_domain TEXT,
                page_count INTEGER
            );
        """)
        for spec in self.filter_table_specs.values():
            self.cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {spec.table_name} (
                    pdf_name TEXT,
                    {spec.value_column} TEXT
                );
                """
            )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_vectors (
                vector_store_key TEXT,
                pdf_name TEXT,
                page TEXT,
                vector BLOB,
                vector_dim INTEGER,
                PRIMARY KEY (vector_store_key, pdf_name, page)
            );
            """
        )
        self.conn.commit()

    def add_batch(self, metadata_dicts):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
    
        for spec in self.filter_table_specs.values():
            self.cursor.execute(
                f"DROP INDEX IF EXISTS idx_{spec.table_name}_value_pdf;"
            )
            self.cursor.execute(f"DROP INDEX IF EXISTS idx_{spec.table_name}_pdf;")

        to_insert = [
            (
                md.get("crawl_url", ""),
                self._normalize_crawl_date(md.get("crawl_date", "")),
                md.get("digest", ""),
                md.get("pretty_name", ""),
                md.get("sub_domain", ""),
                md.get("page_count", 0),
            )
            for md in metadata_dicts
        ]
        self.cursor.executemany(
            "INSERT INTO metadata ("
            "crawl_url, crawl_date, digest, pretty_name, sub_domain, page_count"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            to_insert,
        )

        for spec in self.filter_table_specs.values():
            filter_rows = []
            for md in metadata_dicts:
                pdf_name = md.get("pdf_name", "")
                value = self._normalize_value(spec.field_name, md.get(spec.field_name))
                if not pdf_name or value is None:
                    continue
                filter_rows.append((pdf_name, str(value)))
            if filter_rows:
                self.cursor.executemany(
                    f"INSERT INTO {spec.table_name} (pdf_name, {spec.value_column}"
                    f") VALUES (?, ?)",
                    filter_rows,
                )

        self._total_entries = -1
        self._total_documents = -1
        self.conn.commit()

    def load_index(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        if self._total_entries == -1:
            self.cursor.execute("SELECT COUNT(*) FROM metadata")
            self._total_entries = self.cursor.fetchone()[0]
        self._total_documents = -1

    def save_index(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        for spec in self.filter_table_specs.values():
            self.cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{spec.table_name}_value_pdf "
                f"ON {spec.table_name} ({spec.value_column}, pdf_name);"
            )
            self.cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{spec.table_name}_pdf "
                f"ON {spec.table_name} (pdf_name);"
            )
        self.cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metadata_vectors_key_pdf
            ON metadata_vectors (vector_store_key, pdf_name);
            """
        )

        self.cursor.execute("""PRAGMA journal_mode = WAL;""")
        self.cursor.execute("""PRAGMA optimize;""")

        self.conn.commit()

    def search(self, digests, predicates: list[Predicate] | None = None):
        cursor = self.conn.cursor()
        placeholders = ",".join(["?"] * len(digests))
        query = (
            "SELECT crawl_url, crawl_date, digest, pretty_name, sub_domain, page_count "
            f"FROM metadata WHERE digest IN ({placeholders})"
        )
        params = list(digests)
        if predicates:
            for predicate in predicates:
                _, clauses, clause_params = self._predicate_sql(predicate)
                if clauses:
                    query += " AND " + " AND ".join(clauses)
                    params.extend(clause_params)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        metadata = {}
        for row in rows:
            digest = row[2]
            row_dict = {
                "crawl_url": row[0],
                "crawl_date": row[1],
                "digest": row[2],
                "pretty_name": row[3],
                "sub_domain": row[4],
                "page_count": row[5],
            }
            if digest not in metadata:
                metadata[digest] = [row_dict]
            else:
                metadata[digest].append(row_dict)
        # Return as a dict with lists of dicts representing every time the pdf was
        # crawled.
        return metadata

    def total_entries(self):
        if self._total_entries == -1:
            self.load_index()
        return self._total_entries

    def estimate_selectivity(self, predicates: list[Predicate] | None = None) -> float:
        if not predicates:
            return 1.0
        total_docs = self._total_documents_count()
        if total_docs <= 0:
            return 0.0
        selectivity = 1.0
        for predicate in predicates:
            matching_docs = self._count_pdf_names_for_predicate(predicate)
            selectivity *= matching_docs / total_docs
        return max(0.0, min(1.0, selectivity))

    def get_candidate_pdf_names(
        self, predicates: list[Predicate] | None = None
    ) -> set[str]:
        if not predicates:
            return set()
        candidate_sets = [
            self._query_pdf_names_for_predicate(predicate) for predicate in predicates
        ]
        if not candidate_sets:
            return set()
        candidates = candidate_sets[0]
        for candidate_set in candidate_sets[1:]:
            candidates = candidates.intersection(candidate_set)
        return candidates

    def upsert_vectors(self, vector_store_key, vectors, pdf_names, pages):
        if self.conn is None:
            self.load_index()
        rows = []
        for vector, pdf_name, page in zip(vectors, pdf_names, pages, strict=False):
            if not pdf_name:
                continue
            vector_blob, vector_dim = self._serialize_vector(vector)
            rows.append(
                (
                    str(vector_store_key),
                    str(pdf_name),
                    str(page),
                    sqlite3.Binary(vector_blob),
                    vector_dim,
                )
            )
        if not rows:
            return

        self.cursor.executemany(
            """
            INSERT OR REPLACE INTO metadata_vectors
            (vector_store_key, pdf_name, page, vector, vector_dim)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def get_vectors_for_pdf_names(self, vector_store_key, candidate_pdf_names):
        if self.conn is None:
            self.load_index()
        if not candidate_pdf_names:
            return np.empty((0, 0), dtype=np.float32), [], []

        all_rows = []
        candidate_list = list(candidate_pdf_names)
        chunk_size = 800
        for i in range(0, len(candidate_list), chunk_size):
            chunk = candidate_list[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            query = (
                "SELECT pdf_name, page, vector, vector_dim "
                "FROM metadata_vectors "
                f"WHERE vector_store_key = ? AND pdf_name IN ({placeholders})"
            )
            params = [str(vector_store_key), *chunk]
            all_rows.extend(self.conn.execute(query, params).fetchall())

        if not all_rows:
            return np.empty((0, 0), dtype=np.float32), [], []

        names = []
        pages = []
        vectors = []
        for pdf_name, page, vector_blob, vector_dim in all_rows:
            names.append(pdf_name)
            pages.append(str(page))
            vectors.append(self._deserialize_vector(vector_blob, int(vector_dim)))

        return np.vstack(vectors), names, pages

    def vector_count(self, vector_store_key: str) -> int:
        if self.conn is None:
            self.load_index()
        row = self.conn.execute(
            "SELECT COUNT(*) FROM metadata_vectors WHERE vector_store_key = ?",
            (str(vector_store_key),),
        ).fetchone()
        return int(row[0]) if row else 0


class DuckDBMetadataIndex(AbstractMetadataIndex):
    def __init__(self, index_metadata_directory, filter_table_specs=None):
        self.index_metadata_directory = index_metadata_directory
        self.db_path = os.path.join(self.index_metadata_directory, "metadata.duckdb")
        self.conn = None
        self._total_entries = -1
        self._total_documents = -1
        self.filter_table_specs = (
            filter_table_specs
            if filter_table_specs is not None
            else get_default_filter_specs()
        )

    def _normalize_value(self, field_name, value):
        if value is None:
            return None
        if field_name == "crawl_date":
            return self._normalize_crawl_date(str(value))
        return value

    def _predicate_sql(self, predicate):
        spec = self.filter_table_specs.get(predicate.field_name)
        if spec is None:
            raise ValueError(
                f"Unsupported metadata predicate field: {predicate.field_name}"
            )

        clauses = []
        params = []
        if isinstance(predicate, EqualityPredicate):
            if not spec.supports_exact:
                raise ValueError(
                    f"Field '{predicate.field_name}' does not support exact predicates"
                )
            clauses.append(f"{spec.value_column} = ?")
            params.append(self._normalize_value(predicate.field_name, predicate.value))
        elif isinstance(predicate, RangePredicate):
            if not spec.supports_range:
                raise ValueError(
                    f"Field '{predicate.field_name}' does not support range predicates"
                )
            if predicate.min_val is not None:
                clauses.append(f"{spec.value_column} >= ?")
                params.append(
                    self._normalize_value(predicate.field_name, predicate.min_val)
                )
            if predicate.max_val is not None:
                clauses.append(f"{spec.value_column} <= ?")
                params.append(
                    self._normalize_value(predicate.field_name, predicate.max_val)
                )
        else:
            raise TypeError(f"Unsupported predicate type: {type(predicate)}")

        return spec, clauses, params

    def _query_pdf_names_for_predicate(self, predicate):
        spec, clauses, params = self._predicate_sql(predicate)
        query = f"SELECT DISTINCT pdf_name FROM {spec.table_name}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        rows = self.conn.execute(query, params).fetchall()
        return {row[0] for row in rows}

    def _count_pdf_names_for_predicate(self, predicate):
        spec, clauses, params = self._predicate_sql(predicate)
        query = f"SELECT COUNT(DISTINCT pdf_name) FROM {spec.table_name}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        row = self.conn.execute(query, params).fetchone()
        return row[0] if row else 0

    def _total_documents_count(self):
        if self._total_documents != -1:
            return self._total_documents
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT pdf_name) FROM metadata"
        ).fetchone()
        self._total_documents = row[0] if row else 0
        return self._total_documents

    def _connect(self):
        if self.conn is None:
            os.makedirs(self.index_metadata_directory, exist_ok=True)
            self.conn = duckdb.connect(self.db_path)

    def build_index(self):
        self._connect()
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                crawl_url TEXT,
                crawl_date TEXT,
                digest TEXT,
                pretty_name TEXT,
                sub_domain TEXT,
                page_count INTEGER
            );
        """)
        for spec in self.filter_table_specs.values():
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {spec.table_name} (
                    pdf_name TEXT,
                    {spec.value_column} TEXT
                );
                """
            )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_vectors (
                vector_store_key TEXT,
                pdf_name TEXT,
                page TEXT,
                vector FLOAT[]
            );
            """
        )

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
                "digest": [md.get("digest", "") for md in metadata_dicts],
                "pretty_name": [md.get("pretty_name", "") for md in metadata_dicts],
                "sub_domain": [md.get("sub_domain", "") for md in metadata_dicts],
                "page_count": pa.array(
                    [md.get("page_count", 0) for md in metadata_dicts], type=pa.int32()
                ),
            }
        )
        self.conn.register("_batch", arrow_table)
        self.conn.execute("INSERT INTO metadata SELECT * FROM _batch")
        self.conn.unregister("_batch")

        for spec in self.filter_table_specs.values():
            filter_rows = []
            for md in metadata_dicts:
                pdf_name = md.get("pdf_name", "")
                value = self._normalize_value(spec.field_name, md.get(spec.field_name))
                if not pdf_name or value is None:
                    continue
                filter_rows.append(
                    {
                        "pdf_name": pdf_name,
                        spec.value_column: str(value),
                    }
                )
            if filter_rows:
                filter_table = pa.Table.from_pylist(filter_rows)
                self.conn.register("_filter_batch", filter_table)
                self.conn.execute(
                    f"INSERT INTO {spec.table_name} SELECT * FROM _filter_batch"
                )
                self.conn.unregister("_filter_batch")

        self._total_entries = -1
        self._total_documents = -1

    def load_index(self):
        self._connect()
        if self._total_entries == -1:
            result = self.conn.execute("SELECT COUNT(*) FROM metadata").fetchone()
            self._total_entries = result[0] if result else 0
        self._total_documents = -1

    def save_index(self):
        self._connect()
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_digest ON metadata (digest);
                            """)
        for spec in self.filter_table_specs.values():
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{spec.table_name}_value_pdf "
                f"ON {spec.table_name} ({spec.value_column}, pdf_name);"
            )
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{spec.table_name}_pdf "
                f"ON {spec.table_name} (pdf_name);"
            )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metadata_vectors_key_pdf
            ON metadata_vectors (vector_store_key, pdf_name);
            """
        )
        self.conn.checkpoint()

    def search(self, digests: list[str], predicates: list[Predicate] | None = None):
        self._connect()
        placeholders = ", ".join(["?"] * len(digests))
        query = (
            "SELECT crawl_url, crawl_date, digest, pretty_name, sub_domain, page_count "
            f"FROM metadata WHERE digest IN ({placeholders})"
        )
        params: list = list(digests)
        if predicates:
            for predicate in predicates:
                _, clauses, clause_params = self._predicate_sql(predicate)
                if clauses:
                    query += " AND " + " AND ".join(clauses)
                    params.extend(clause_params)
        rows = self.conn.execute(query, params).fetchall()
        metadata = {}
        for row in rows:
            digest = row[2]
            row_dict = {
                "crawl_url": row[0],
                "crawl_date": row[1],
                "digest": row[2],
                "pretty_name": row[3],
                "sub_domain": row[4],
                "page_count": row[5],
            }
            if digest not in metadata:
                metadata[digest] = [row_dict]
            else:
                metadata[digest].append(row_dict)
        # Return as a dict with lists of dicts representing every time the pdf was
        # crawled.
        return metadata

    def total_entries(self):
        if self._total_entries == -1:
            self.load_index()
        return self._total_entries

    def estimate_selectivity(self, predicates: list[Predicate] | None = None) -> float:
        if not predicates:
            return 1.0
        total_docs = self._total_documents_count()
        if total_docs <= 0:
            return 0.0
        selectivity = 1.0
        for predicate in predicates:
            matching_docs = self._count_pdf_names_for_predicate(predicate)
            selectivity *= matching_docs / total_docs
        return max(0.0, min(1.0, selectivity))

    def get_candidate_pdf_names(
        self, predicates: list[Predicate] | None = None
    ) -> set[str]:
        if not predicates:
            return set()
        candidate_sets = [
            self._query_pdf_names_for_predicate(predicate) for predicate in predicates
        ]
        if not candidate_sets:
            return set()
        candidates = candidate_sets[0]
        for candidate_set in candidate_sets[1:]:
            candidates = candidates.intersection(candidate_set)
        return candidates

    def upsert_vectors(self, vector_store_key, vectors, pdf_names, pages):
        self._connect()
        rows = []
        for vector, pdf_name, page in zip(vectors, pdf_names, pages, strict=False):
            if not pdf_name:
                continue
            rows.append(
                {
                    "vector_store_key": str(vector_store_key),
                    "pdf_name": str(pdf_name),
                    "page": str(page),
                    "vector": np.asarray(vector, dtype=np.float32).tolist(),
                }
            )
        if not rows:
            return

        vector_table = pa.Table.from_pylist(rows)
        self.conn.register("_vector_batch", vector_table)
        self.conn.execute(
            """
            DELETE FROM metadata_vectors
            USING _vector_batch
            WHERE metadata_vectors.vector_store_key = _vector_batch.vector_store_key
              AND metadata_vectors.pdf_name = _vector_batch.pdf_name
              AND metadata_vectors.page = _vector_batch.page
            """
        )
        self.conn.execute("INSERT INTO metadata_vectors SELECT * FROM _vector_batch")
        self.conn.unregister("_vector_batch")

    def get_vectors_for_pdf_names(self, vector_store_key, candidate_pdf_names):
        self._connect()
        if not candidate_pdf_names:
            return np.empty((0, 0), dtype=np.float32), [], []

        candidate_list = list(candidate_pdf_names)
        rows = []
        chunk_size = 800
        for i in range(0, len(candidate_list), chunk_size):
            chunk = candidate_list[i : i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            query = (
                "SELECT pdf_name, page, vector FROM metadata_vectors "
                f"WHERE vector_store_key = ? AND pdf_name IN ({placeholders})"
            )
            params = [str(vector_store_key), *chunk]
            rows.extend(self.conn.execute(query, params).fetchall())

        if not rows:
            return np.empty((0, 0), dtype=np.float32), [], []

        names = [row[0] for row in rows]
        pages = [str(row[1]) for row in rows]
        vectors = np.asarray([row[2] for row in rows], dtype=np.float32)
        return vectors, names, pages

    def vector_count(self, vector_store_key: str) -> int:
        self._connect()
        row = self.conn.execute(
            "SELECT COUNT(*) FROM metadata_vectors WHERE vector_store_key = ?",
            [str(vector_store_key)],
        ).fetchone()
        return int(row[0]) if row else 0
