# AI modified
"""Metadata index implementations used by serving and filtering planners."""

import os
import sqlite3
import threading
from abc import ABC, abstractmethod

import duckdb
import pyarrow as pa

from ..query import EqualityPredicate, Predicate, RangePredicate
from .base import AbstractIndex

# See SQLiteMetadataIndex._read_connection.
_INHERITED_CONNECTIONS: list[sqlite3.Connection] = []


class AbstractMetadataIndex(AbstractIndex, ABC):
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
    def get_candidate_digests(
        self, predicates: list[Predicate] | None = None
    ) -> set[str]:
        """Return distinct digests that satisfy all predicates."""

    @staticmethod
    def _normalize_crawl_date(date_str: str) -> str:
        """Truncate crawl_date to YYYYMMDD, stripping any trailing time component."""
        return date_str.replace("-", "")[:8]


class SQLiteMetadataIndex(AbstractMetadataIndex):
    def __init__(self, index_metadata_directory):
        self.index_metadata_directory = index_metadata_directory
        self.db_path = os.path.join(self.index_metadata_directory, "metadata.db")
        # self.conn is used for building/writing the index (single threaded).
        # Searches use _read_connection, which is per thread and per process.
        self.conn = None
        self.cursor = None
        self._local = threading.local()
        self._total_entries = -1
        self._total_documents = -1

    def _predicate_sql(self, predicate):
        fn = predicate.field_name
        if fn == "sub_domain":
            table, col = "metadata_sub_domain", "sub_domain"
        elif fn == "crawl_date":
            table, col = "metadata_crawl_date", "crawl_date"
        else:
            raise ValueError(f"Unsupported metadata predicate field: {fn}")

        clauses, params = [], []
        if isinstance(predicate, EqualityPredicate):
            val = (
                self._normalize_crawl_date(str(predicate.value))
                if fn == "crawl_date"
                else predicate.value
            )
            clauses.append(f"{col} = ?")
            params.append(val)
        elif isinstance(predicate, RangePredicate):
            if fn == "sub_domain":
                raise ValueError("sub_domain does not support range predicates")
            if predicate.min_val is not None:
                clauses.append(f"{col} >= ?")
                params.append(self._normalize_crawl_date(str(predicate.min_val)))
            if predicate.max_val is not None:
                clauses.append(f"{col} <= ?")
                params.append(self._normalize_crawl_date(str(predicate.max_val)))
        else:
            raise TypeError(f"Unsupported predicate type: {type(predicate)}")

        return table, clauses, params

    def _read_connection(self) -> sqlite3.Connection:
        # sqlite3 connections must not be used concurrently from several threads
        # (hybrid search queries in parallel, and gunicorn workers are threaded)
        # nor across fork() (the server is loaded before gunicorn forks).
        conn = getattr(self._local, "conn", None)
        if conn is None or self._local.pid != os.getpid():
            if conn is not None:
                # Keep the handle inherited from the parent alive: closing it in
                # the child (e.g. via garbage collection) can release the
                # parent's locks.
                _INHERITED_CONNECTIONS.append(conn)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn = conn
            self._local.pid = os.getpid()
        return conn

    def _query_digests_for_predicate(self, predicate):
        table, clauses, params = self._predicate_sql(predicate)
        query = f"SELECT DISTINCT digest FROM {table}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        cursor = self._read_connection().cursor()
        cursor.execute(query, params)
        return {row[0] for row in cursor.fetchall()}

    def _count_digests_for_predicate(self, predicate):
        table, clauses, params = self._predicate_sql(predicate)
        query = f"SELECT COUNT(DISTINCT digest) FROM {table}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        cursor = self._read_connection().cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        return row[0] if row else 0

    def _total_documents_count(self):
        if self._total_documents != -1:
            return self._total_documents
        cursor = self._read_connection().cursor()
        cursor.execute("SELECT COUNT(DISTINCT digest) FROM metadata")
        row = cursor.fetchone()
        self._total_documents = row[0] if row else 0
        return self._total_documents

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
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_sub_domain (
                digest TEXT,
                sub_domain TEXT
            );
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_crawl_date (
                digest TEXT,
                crawl_date TEXT
            );
            """
        )
        self.conn.commit()

    def add_batch(self, metadata_dicts):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()

        self.cursor.execute("DROP INDEX IF EXISTS idx_metadata_sub_domain_value_pdf;")
        self.cursor.execute("DROP INDEX IF EXISTS idx_metadata_sub_domain_pdf;")
        self.cursor.execute("DROP INDEX IF EXISTS idx_metadata_crawl_date_value_pdf;")
        self.cursor.execute("DROP INDEX IF EXISTS idx_metadata_crawl_date_pdf;")

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

        sub_domain_rows = [
            (md.get("digest", ""), md.get("sub_domain", ""))
            for md in metadata_dicts
            if md.get("digest") and md.get("sub_domain") is not None
        ]
        if sub_domain_rows:
            self.cursor.executemany(
                "INSERT INTO metadata_sub_domain (digest, sub_domain) VALUES (?, ?)",
                sub_domain_rows,
            )

        crawl_date_rows = [
            (md.get("digest", ""), self._normalize_crawl_date(md.get("crawl_date", "")))
            for md in metadata_dicts
            if md.get("digest") and md.get("crawl_date") is not None
        ]
        if crawl_date_rows:
            self.cursor.executemany(
                "INSERT INTO metadata_crawl_date (digest, crawl_date) VALUES (?, ?)",
                crawl_date_rows,
            )

        self._total_entries = -1
        self._total_documents = -1
        self.conn.commit()

    def load_index(self):
        if self._total_entries == -1:
            cursor = self._read_connection().cursor()
            cursor.execute("SELECT COUNT(*) FROM metadata")
            self._total_entries = cursor.fetchone()[0]
        self._total_documents = -1

    def save_index(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_sub_domain_sub_domain_digest "
            "ON metadata_sub_domain (digest, sub_domain);"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_crawl_date_crawl_date_digest "
            "ON metadata_crawl_date (digest, crawl_date);"
        )

        self.cursor.execute("""PRAGMA journal_mode = WAL;""")
        self.cursor.execute("""PRAGMA optimize;""")

        self.conn.commit()

    def search(self, digests, predicates: list[Predicate] | None = None):
        cursor = self._read_connection().cursor()
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
            matching_docs = self._count_digests_for_predicate(predicate)
            selectivity *= matching_docs / total_docs
        return max(0.0, min(1.0, selectivity))

    def get_candidate_digests(
        self, predicates: list[Predicate] | None = None
    ) -> set[str]:
        if not predicates:
            return set()
        candidate_sets = [
            self._query_digests_for_predicate(predicate) for predicate in predicates
        ]
        if not candidate_sets:
            return set()
        candidates = candidate_sets[0]
        for candidate_set in candidate_sets[1:]:
            candidates = candidates.intersection(candidate_set)
        return candidates


class DuckDBMetadataIndex(AbstractMetadataIndex):
    def __init__(self, index_metadata_directory):
        self.index_metadata_directory = index_metadata_directory
        self.db_path = os.path.join(self.index_metadata_directory, "metadata.duckdb")
        self.conn = None
        self._local = threading.local()
        self._total_entries = -1
        self._total_documents = -1

    def _predicate_sql(self, predicate):
        fn = predicate.field_name
        if fn == "sub_domain":
            table, col = "metadata_sub_domain", "sub_domain"
        elif fn == "crawl_date":
            table, col = "metadata_crawl_date", "crawl_date"
        else:
            raise ValueError(f"Unsupported metadata predicate field: {fn}")

        clauses, params = [], []
        if isinstance(predicate, EqualityPredicate):
            val = (
                self._normalize_crawl_date(str(predicate.value))
                if fn == "crawl_date"
                else predicate.value
            )
            clauses.append(f"{col} = ?")
            params.append(val)
        elif isinstance(predicate, RangePredicate):
            if fn == "sub_domain":
                raise ValueError("sub_domain does not support range predicates")
            if predicate.min_val is not None:
                clauses.append(f"{col} >= ?")
                params.append(self._normalize_crawl_date(str(predicate.min_val)))
            if predicate.max_val is not None:
                clauses.append(f"{col} <= ?")
                params.append(self._normalize_crawl_date(str(predicate.max_val)))
        else:
            raise TypeError(f"Unsupported predicate type: {type(predicate)}")

        return table, clauses, params

    def _query_digests_for_predicate(self, predicate):
        table, clauses, params = self._predicate_sql(predicate)
        query = f"SELECT DISTINCT digest FROM {table}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        return {
            row[0] for row in self._read_connection().execute(query, params).fetchall()
        }

    def _count_digests_for_predicate(self, predicate):
        table, clauses, params = self._predicate_sql(predicate)
        query = f"SELECT COUNT(DISTINCT digest) FROM {table}"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        row = self._read_connection().execute(query, params).fetchone()
        return row[0] if row else 0

    def _total_documents_count(self):
        if self._total_documents != -1:
            return self._total_documents
        row = (
            self._read_connection()
            .execute("SELECT COUNT(DISTINCT digest) FROM metadata")
            .fetchone()
        )
        self._total_documents = row[0] if row else 0
        return self._total_documents

    def _connect(self):
        if self.conn is None:
            os.makedirs(self.index_metadata_directory, exist_ok=True)
            self.conn = duckdb.connect(self.db_path)

    def _read_connection(self) -> duckdb.DuckDBPyConnection:
        # A DuckDB connection must not be used concurrently from several threads
        # (hybrid search queries in parallel); each thread reads through its own
        # cursor of the shared connection.
        cursor = getattr(self._local, "cursor", None)
        if cursor is None:
            self._connect()
            cursor = self.conn.cursor()
            self._local.cursor = cursor
        return cursor

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
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_sub_domain (
                digest TEXT,
                sub_domain TEXT
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_crawl_date (
                digest TEXT,
                crawl_date TEXT
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

        sub_domain_rows = [
            {"digest": md.get("digest", ""), "sub_domain": md.get("sub_domain", "")}
            for md in metadata_dicts
            if md.get("digest") and md.get("sub_domain") is not None
        ]
        if sub_domain_rows:
            tbl = pa.Table.from_pylist(sub_domain_rows)
            self.conn.register("_filter_batch", tbl)
            self.conn.execute(
                "INSERT INTO metadata_sub_domain SELECT * FROM _filter_batch"
            )
            self.conn.unregister("_filter_batch")

        crawl_date_rows = [
            {
                "digest": md.get("digest", ""),
                "crawl_date": self._normalize_crawl_date(md.get("crawl_date", "")),
            }
            for md in metadata_dicts
            if md.get("digest") and md.get("crawl_date") is not None
        ]
        if crawl_date_rows:
            tbl = pa.Table.from_pylist(crawl_date_rows)
            self.conn.register("_filter_batch", tbl)
            self.conn.execute(
                "INSERT INTO metadata_crawl_date SELECT * FROM _filter_batch"
            )
            self.conn.unregister("_filter_batch")

        self._total_entries = -1
        self._total_documents = -1

    def load_index(self):
        self._connect()
        if self._total_entries == -1:
            result = (
                self._read_connection()
                .execute("SELECT COUNT(*) FROM metadata")
                .fetchone()
            )
            self._total_entries = result[0] if result else 0
        self._total_documents = -1

    def save_index(self):
        self._connect()
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_digest ON metadata (digest);
                            """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_sub_domain_value_digest "
            "ON metadata_sub_domain (sub_domain, digest);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_sub_domain_digest "
            "ON metadata_sub_domain (digest);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_crawl_date_value_digest "
            "ON metadata_crawl_date (crawl_date, digest);"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metadata_crawl_date_digest "
            "ON metadata_crawl_date (digest);"
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
        rows = self._read_connection().execute(query, params).fetchall()
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
            matching_docs = self._count_digests_for_predicate(predicate)
            selectivity *= matching_docs / total_docs
        return max(0.0, min(1.0, selectivity))

    def get_candidate_digests(
        self, predicates: list[Predicate] | None = None
    ) -> set[str]:
        if not predicates:
            return set()
        candidate_sets = [
            self._query_digests_for_predicate(predicate) for predicate in predicates
        ]
        if not candidate_sets:
            return set()
        candidates = candidate_sets[0]
        for candidate_set in candidate_sets[1:]:
            candidates = candidates.intersection(candidate_set)
        return candidates
