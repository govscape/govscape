# AI modified: 2026-03-08 f62d40b8
# AI modified: 2026-03-14 4a6b1b72
# This file defines the logic for serving requests to the user.
import math
import os
import time

import boto3
from flask import Flask
from flask_cors import CORS

from .api import init_api
from .config import ServerConfig
from .indexing import (
    AbstractKeywordIndex,
    AbstractVectorIndex,
    FAISSIndex,
    LanceDBKeywordIndex,
    LuceneKeywordIndex,
    SQLiteKeywordIndex,
    SQLiteMetadataIndex,
    WhooshKeywordIndex,
)
from .query import Query, Response


# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize
#    FAISS to do this
class Server:
    # obtain all the setup information from configuration
    def __init__(self, config: ServerConfig):
        self.config = config

        # Directories
        self.metadata_directory = config.metadata_directory
        self.embedding_directory = config.embedding_directory
        self.embedding_img_pg_directory = config.embedding_img_pg_directory
        self.index_directory = config.index_directory
        self.index_img_pg_directory = config.index_img_pg_directory
        self.index_keyword_directory = config.index_keyword_directory
        self.index_metadata_directory = config.index_metadata_directory
        self.image_directory = config.image_directory
        self.stats_file = config.stats_file
        self.blacklist_file = config.blacklist_file
        self.vector_index_type = config.vector_index_type
        self.keyword_index_type = config.keyword_index_type
        self.k = config.k
        self.max_crawl_instances = config.max_crawl_instances

        # Index configuration
        self.index_config = config.index_config

        # Model Params
        self.text_model = config.text_model
        self.text_d = config.text_d

        self.visual_model = config.visual_model
        self.visual_d = config.visual_d

        if self.vector_index_type == "Memory":
            self.text_index = FAISSIndex(self.index_directory)
            self.visual_index = FAISSIndex(self.index_img_pg_directory)
        else:
            raise ValueError(f"Unsupported vector index type: {self.vector_index_type}")
        self.text_index.load_index()
        self.visual_index.load_index()

        if self.keyword_index_type == "LanceDB":
            self.keyword_index: AbstractKeywordIndex = LanceDBKeywordIndex(
                self.index_keyword_directory
            )
        elif self.keyword_index_type == "SQLite":
            self.keyword_index = SQLiteKeywordIndex(self.index_keyword_directory)
        elif self.keyword_index_type == "Whoosh":
            self.keyword_index = WhooshKeywordIndex(self.index_keyword_directory)
        elif self.keyword_index_type == "Lucene":
            self.keyword_index = LuceneKeywordIndex(self.index_keyword_directory)
        else:
            raise ValueError(
                f"Unsupported keyword index type: {self.keyword_index_type}"
            )
        self.keyword_index.load_index()

        self.metadata_index = SQLiteMetadataIndex(self.index_metadata_directory)
        self.metadata_index.load_index()

        self.blacklist: set[str] = self._load_blacklist()

        self.s3 = boto3.client("s3")

        # Get the absolute path to the build directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.abspath(
            os.path.join(current_dir, "..", "..", "interface", "build")
        )

        print(f"Static files directory: {build_dir}")
        if not os.path.exists(build_dir):
            print(f"Warning: Build directory does not exist: {build_dir}")
            print("Please run 'npm run build' in the interface directory first")

        # Initialize Flask app and API
        self.app = Flask(__name__, static_folder=build_dir, static_url_path="")
        # TODO: Remove localhost:5173 soon for security concerns
        CORS(
            self.app,
            origins=["https://govscape.net", "http://localhost:5173"],
            supports_credentials=True,
        )

        @self.app.route("/")
        def serve_index():
            print("Serving index.html")
            return self.app.send_static_file("index.html")

        self.app.server = self  # type: ignore[attr-defined]
        self.api = init_api(self.app)

    def _load_blacklist(self) -> set[str]:
        path = self.blacklist_file
        if not os.path.exists(path):
            print(f"No blacklist file at {path}; starting with empty blacklist")
            return set()
        try:
            with open(path, encoding="utf-8") as f:
                entries = {
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                }
            print(f"Loaded {len(entries)} blacklist entries")
            return entries
        except OSError as e:
            print(f"Warning: failed to read blacklist at {path}: {e}")
            return set()

    @staticmethod
    def deduplicate_responses(D, names, pages):
        seen = set()
        unique_distances = []
        unique_names = []
        unique_pages = []
        for distance, name, page in zip(D, names, pages, strict=False):
            if name not in seen:
                seen.add(name)
                unique_distances.append(distance)
                unique_names.append(name)
                unique_pages.append(page)
        return unique_distances, unique_names, unique_pages

    def search(self, query: Query) -> Response:
        search_type = query.search_type
        predicates = query.predicates
        page = query.page
        index: AbstractKeywordIndex | AbstractVectorIndex
        if search_type == "textual":
            query_embedding = self.text_model.encode_text(query.q_text, is_query=True)
            index = self.text_index
        elif search_type == "visual":
            query_embedding = self.visual_model.encode_text(query.q_text)
            index = self.visual_index
        elif search_type == "keyword":
            query_embedding = query.q_text
            index = self.keyword_index
        else:
            raise ValueError(f"Unsupported search type: {search_type}")

        current_k = self.k * 2
        results_needed_for_page = (
            page * self.k + 1
        )  # we need one extra result to check if there is a next page
        search_results: list[dict] = []
        old_results_found = -1
        while len(search_results) < results_needed_for_page:
            # Search for the k closest arrays
            start = time.time()

            D, pdf_names, pdf_pages = index.search(query_embedding, current_k)
            D, pdf_names, pdf_pages = self.deduplicate_responses(
                D, pdf_names, pdf_pages
            )

            if self.blacklist:
                filtered = [
                    (d, n, p)
                    for d, n, p in zip(D, pdf_names, pdf_pages, strict=False)
                    if n not in self.blacklist
                ]
                if filtered:
                    D, pdf_names, pdf_pages = (
                        list(x) for x in zip(*filtered, strict=False)
                    )
                else:
                    D, pdf_names, pdf_pages = [], [], []

            print(f"Index Search took {time.time() - start} seconds")
            print(
                "Search type: "
                f"{search_type}, current_k: {current_k}, "
                f"results found: {len(D)}"
            )
            pdf_metadata = self.metadata_index.search(pdf_names, predicates)
            print(
                "Search type: "
                f"{search_type}, current_k: {current_k}, results found "
                f"after filtering: {len(pdf_metadata)}"
            )
            search_results = []
            for distance, name, page_num in zip(D, pdf_names, pdf_pages, strict=False):
                metadata = pdf_metadata.get(name, None)
                if metadata:
                    all_records = sorted(
                        metadata, key=lambda r: r.get("crawl_date", ""), reverse=True
                    )
                    has_more_crawls = len(all_records) > self.max_crawl_instances
                    limited_records = all_records[: self.max_crawl_instances]
                    newest = all_records[0]
                    jpeg_file = (
                        self.image_directory
                        + "/"
                        + name
                        + "/"
                        + name
                        + "_"
                        + page_num
                        + ".jpeg"
                    )
                    search_results.append(
                        {
                            "pdf": name,
                            "page": page_num,
                            "distance": float(distance),
                            "jpeg": jpeg_file,
                            "crawl_url": newest.get("crawl_url", ""),
                            "crawl_date": newest.get("crawl_date", ""),
                            "sub_domain": newest.get("sub_domain", ""),
                            "has_more_crawls": has_more_crawls,
                            "crawl_instances": [
                                {
                                    "crawl_url": r.get("crawl_url", ""),
                                    "crawl_date": r.get("crawl_date", ""),
                                    "sub_domain": r.get("sub_domain", ""),
                                }
                                for r in limited_records
                            ],
                        }
                    )
            if current_k > min(100000, index.total_entries()):
                # TODO: If we have to expand beyond 100k, we should simply do
                # the filtering first.
                break

            if len(search_results) >= results_needed_for_page:
                break  # If we have enough results for our target page, we can stop.

            results_found = len(search_results)
            if results_found == old_results_found and (len(predicates) == 0):
                break  # No more results can be found even after increasing k
            old_results_found = results_found

            current_k *= 2  # Double the k until we have enough results

        start_index = (page - 1) * self.k
        end_index = start_index + self.k

        total_count = self._get_total_pdfs_count() or 0
        total_pages = math.ceil(total_count / self.k) if total_count else 0

        return Response(
            results=search_results[start_index:end_index],
            pagination={
                "page": page,
                "page_size": self.k,
                "has_next_page": len(search_results) > end_index,
                "total_count": total_count,
                "total_pages": total_pages,
            },
        )

    def pdf_pages(self, pdf_id):
        """
        Get all page images and metadata for a PDF by pdf_id.
        Returns dict with 'images', 'crawl_url', 'crawl_date', 'sub_domain'
        (newest crawl), and 'crawl_instances' (all crawls, newest first).
        """
        if not pdf_id:
            return {"error": "Missing 'pdf_id' parameter"}, 400

        if pdf_id in self.blacklist:
            return {
                "images": [],
                "crawl_url": "",
                "crawl_date": "",
                "sub_domain": "",
                "has_more_crawls": False,
                "crawl_instances": [],
            }

        md = self.metadata_index.search([pdf_id]) or {}
        records = md.get(pdf_id) or [{}]

        # Sort all crawl records newest-first; crawl_date is YYYY-MM-DD so
        # lexicographic descending sort is correct.
        records = sorted(records, key=lambda r: r.get("crawl_date", ""), reverse=True)
        has_more_crawls = len(records) > self.max_crawl_instances
        limited_records = records[: self.max_crawl_instances]
        newest = records[0]

        crawl_url = newest.get("crawl_url", "")
        crawl_date = newest.get("crawl_date", "")
        sub_domain = newest.get("sub_domain", "")
        page_count = int(newest.get("page_count", 0))

        image_dir = os.path.join(self.image_directory, pdf_id)
        images = [f"{image_dir}/{pdf_id}_{i}.jpeg" for i in range(page_count)]

        crawl_instances = [
            {
                "crawl_url": r.get("crawl_url", ""),
                "crawl_date": r.get("crawl_date", ""),
                "sub_domain": r.get("sub_domain", ""),
            }
            for r in limited_records
        ]

        return {
            "images": images,
            "crawl_url": crawl_url,
            "crawl_date": crawl_date,
            "sub_domain": sub_domain,
            "has_more_crawls": has_more_crawls,
            "crawl_instances": crawl_instances,
        }

    def _get_total_pdfs_count(self):
        if hasattr(self, "_total_pdfs_cache"):
            return self._total_pdfs_cache

        total_pdfs_path = self.stats_file

        if not total_pdfs_path or not os.path.exists(total_pdfs_path):
            return 0

        try:
            with open(total_pdfs_path, encoding="utf-8") as f:
                content = f.read().strip()
                self._total_pdfs_cache = int(content)
                return self._total_pdfs_cache

        except Exception as e:
            print(f"Error reading total_pdfs.txt: {str(e)}")
            self._total_pdfs_cache = 0
            return 0

    def run(self, host="localhost", port=8080, debug=False):
        """Run the Flask server."""
        self.app.run(host=host, port=port, debug=debug, threaded=True)
