# AI modified: 2026-04-20 00:00:00 c1b6021e
# AI modified: 2026-04-20 00:00:00 c1b6021e
import contextlib
import os
import pickle as pkl
import sys
from abc import ABC, abstractmethod

import numpy as np

import pyarrow as pa
from lancedb import connect


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
