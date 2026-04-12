import os
import sqlite3
from abc import ABC, abstractmethod

import duckdb
import pyarrow as pa

from ..query import EqualityPredicate, Predicate, RangePredicate


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

    @staticmethod
    def _normalize_crawl_date(date_str: str) -> str:
        """Truncate crawl_date to YYYYMMDD, stripping any trailing time component."""
        return date_str[:8]


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
        self.conn.execute("PRAGMA journal_mode = OFF;")
        self.conn.execute("PRAGMA synchronous = OFF;")
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
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
        self.cursor.execute("DROP INDEX IF EXISTS idx_pdf_name_sub_domain_crawl_date;")
        to_insert = [
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
        self.cursor.executemany(
            "INSERT INTO metadata ("
            "crawl_url, crawl_date, pdf_name, sub_domain, page_count, s3_url"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            to_insert,
        )
        self.conn.commit()

    def load_index(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        if self._total_entries == -1:
            self.cursor.execute("SELECT COUNT(*) FROM metadata")
            self._total_entries = self.cursor.fetchone()[0]

    def save_index(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Set of indexes to speed up:
        # 1. pdf_name
        # 2. pdf_name + crawl_date # Adding a separate index here does not help
        # 3. pdf_name + sub_domain
        # 4. pdf_name + sub_domain + crawl_date

        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pdf_name_sub_domain_crawl_date
            ON metadata (pdf_name, sub_domain, crawl_date);""")

        self.cursor.execute("""PRAGMA journal_mode = WAL;""")
        self.cursor.execute("""PRAGMA optimize;""")

        self.conn.commit()

    def search(self, pdf_names, predicates: list[Predicate] | None = None):
        cursor = self.conn.cursor()
        placeholders = ",".join(["?"] * len(pdf_names))
        query = (
            "SELECT crawl_url, crawl_date, pdf_name, sub_domain, s3_url, page_count "
            f"FROM metadata WHERE pdf_name IN ({placeholders})"
        )
        params = list(pdf_names)
        if predicates:
            for predicate in predicates:
                if isinstance(predicate, EqualityPredicate):
                    query += " AND " + predicate.field_name + " = ?"
                    params.append(predicate.value)
                elif isinstance(predicate, RangePredicate):
                    if predicate.min_val is not None:
                        query += " AND " + predicate.field_name + " >= ?"
                        params.append(predicate.min_val)
                    if predicate.max_val is not None:
                        query += " AND " + predicate.field_name + " <= ?"
                        params.append(predicate.max_val)
        cursor.execute(query, params)
        rows = cursor.fetchall()
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

    def total_entries(self):
        if self._total_entries == -1:
            self.load_index()
        return self._total_entries


class DuckDBMetadataIndex(AbstractMetadataIndex):
    def __init__(self, index_metadata_directory):
        self.index_metadata_directory = index_metadata_directory
        self.db_path = os.path.join(self.index_metadata_directory, "metadata.duckdb")
        self.conn = None
        self._total_entries = -1

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
                pdf_name TEXT,
                sub_domain TEXT,
                page_count INTEGER,
                s3_url TEXT
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
        self.conn.execute("INSERT INTO metadata SELECT * FROM _batch")
        self.conn.unregister("_batch")

    def load_index(self):
        self._connect()
        if self._total_entries == -1:
            result = self.conn.execute("SELECT COUNT(*) FROM metadata").fetchone()
            self._total_entries = result[0] if result else 0

    def save_index(self):
        self._connect()
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pdf_name ON metadata (pdf_name);
                            """)
        self.conn.checkpoint()

    def search(self, pdf_names: list[str], predicates: list[Predicate] | None = None):
        self._connect()
        placeholders = ", ".join(["?"] * len(pdf_names))
        query = (
            "SELECT crawl_url, crawl_date, pdf_name, sub_domain, s3_url, page_count "
            f"FROM metadata WHERE pdf_name IN ({placeholders})"
        )
        params: list = list(pdf_names)
        if predicates:
            for predicate in predicates:
                if isinstance(predicate, EqualityPredicate):
                    query += " AND " + predicate.field_name + " = ?"
                    params.append(predicate.value)
                elif isinstance(predicate, RangePredicate):
                    if predicate.min_val is not None:
                        query += " AND " + predicate.field_name + " >= ?"
                        params.append(predicate.min_val)
                    if predicate.max_val is not None:
                        query += " AND " + predicate.field_name + " <= ?"
                        params.append(predicate.max_val)
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

    def total_entries(self):
        if self._total_entries == -1:
            self.load_index()
        return self._total_entries
