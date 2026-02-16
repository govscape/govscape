# This file defines the logic for serving requests to the user.
from flask import Flask, send_from_directory
from flask_cors import CORS
from .config import ServerConfig
import numpy as np
import os
import boto3
import time
import math
from .api import init_api
from .filter import Filter
from .indexing import DiskANNIndex, FAISSIndex, LanceDBKeywordIndex, SQLiteKeywordIndex, WhooshKeywordIndex, SQLiteMetadataIndex
try:
    # Optional: only available if Lucene library is installed
    from .indexing import LuceneKeywordIndex  # type: ignore
    _HAS_LUCENE = True
except Exception:
    _HAS_LUCENE = False



# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize FAISS to do this
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
        self.vector_index_type = config.vector_index_type
        self.keyword_index_type = config.keyword_index_type
        self.k = config.k

        # Index configuration
        self.index_config = config.index_config

        # Model Params
        self.text_model = config.text_model
        self.text_d = config.text_d

        self.visual_model = config.visual_model
        self.visual_d = config.visual_d

        if self.vector_index_type == 'Disk':
            self.text_index = DiskANNIndex(self.embedding_directory, self.index_directory)
            self.visual_index = DiskANNIndex(self.embedding_directory, self.index_img_pg_directory)
        elif self.vector_index_type == 'Memory':
            self.text_index = FAISSIndex(self.index_directory)
            self.visual_index = FAISSIndex(self.index_img_pg_directory)
        else:
            raise ValueError(f"Unsupported vector index type: {self.vector_index_type}")
        self.text_index.load_index()
        self.visual_index.load_index()

        if self.keyword_index_type == 'LanceDB':
            self.keyword_index = LanceDBKeywordIndex(self.index_keyword_directory)
        elif self.keyword_index_type == 'SQLite':
            self.keyword_index = SQLiteKeywordIndex(self.index_keyword_directory)
        elif self.keyword_index_type == 'Whoosh':
            self.keyword_index = WhooshKeywordIndex(self.index_keyword_directory)
        elif self.keyword_index_type == 'Lucene':
            if not _HAS_LUCENE:
                raise RuntimeError("LuceneKeywordIndex requested but Lucene client is not installed")
            self.keyword_index = LuceneKeywordIndex(self.index_keyword_directory)
        else:
            raise ValueError(f"Unsupported keyword index type: {self.keyword_index_type}")
        self.keyword_index.load_index()

        self.metadata_index = SQLiteMetadataIndex(self.index_metadata_directory)
        self.metadata_index.load_index()

        self.filt = Filter(config)
        self.s3 = boto3.client('s3')

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
        CORS(self.app, origins=["https://govscape.net", "http://localhost:5173"], supports_credentials=True)

        @self.app.route("/")
        def serve_index():
            print("Serving index.html")
            return self.app.send_static_file("index.html")

        self.app.server = self
        self.api = init_api(self.app)

    @staticmethod
    def deduplicate_responses(D, names, pages):
        seen = set()
        unique_distances = []
        unique_names = []
        unique_pages = []
        for distance, name, page in zip(D, names, pages):
            if name not in seen:
                seen.add(name)
                unique_distances.append(distance)
                unique_names.append(name)
                unique_pages.append(page)
        return unique_distances, unique_names, unique_pages

    def search(self, query, search_type='textual', filters=None, page=1):
        if search_type == 'textual':
            query_embedding = self.text_model.encode_text(query, is_query=True)
            index = self.text_index
        elif search_type == 'visual':
            query_embedding = self.visual_model.encode_text(query)
            index = self.visual_index
        elif search_type == 'keyword':
            query_embedding = query
            index = self.keyword_index
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
        
        current_k = self.k * 2
        results_needed_for_page = page * self.k + 1 # we need one extra result to check if there is a next page
        search_results = []
        old_results_found = -1
        while len(search_results) < results_needed_for_page:
            # Search for the k closest arrays
            start = time.time()
            
            D, pdf_names, pdf_pages = index.search(query_embedding, current_k)
            D, pdf_names, pdf_pages = self.deduplicate_responses(D, pdf_names, pdf_pages)
            
            print(f"Index Search took {time.time() - start} seconds")
            print(f"Search type: {search_type}, current_k: {current_k}, results found: {len(D)}")
            pdf_metadata = self.metadata_index.search(pdf_names, filters)
            print(f"Search type: {search_type}, current_k: {current_k}, results found after filtering: {len(pdf_metadata)}")
            search_results = []
            for distance, name, page_num in zip(D, pdf_names, pdf_pages):
                metadata = pdf_metadata.get(name, None)
                if metadata:
                    metadata = metadata[0]  # TODO: Handle multiple crawl dates
                    jpeg_file = self.image_directory + "/" + name + "/" + name + "_" + page_num + '.jpeg'
                    search_results.append({
                        "pdf": name, 
                        "page": page_num,
                        "distance": float(distance), 
                        "jpeg": jpeg_file,
                        "crawl_url" : metadata.get("crawl_url", ""),
                        "crawl_date": metadata.get("crawl_date", ""),
                        "sub_domain": metadata.get("sub_domain", ""),
                    })
            if current_k > min(100000, index.total_entries()): 
                break # TODO: If we have to expand beyond 100k, we should simply do the filtering first

            if len(search_results) >= results_needed_for_page:
                break # If we have enough results for our target page, we can stop.
            
            results_found = len(search_results)
            if results_found == old_results_found and (len(filters or {}) == 0):
                break # No more results can be found even after increasing k
            old_results_found = results_found

            current_k *= 2  # Double the k until we have enough results

        start_index = (page - 1) * self.k
        end_index = start_index + self.k

        total_count = self._get_total_pdfs_count() or 0
        total_pages = math.ceil(total_count / self.k) if total_count else 0

        return {
            "results": search_results[start_index:end_index],
            "pagination": {
                "page": page,
                "page_size": self.k,
                "has_next_page": len(search_results) > end_index,
                "total_count": total_count,
                "total_pages": total_pages,
            },
        }

    def pdf_pages(self, pdf_id):
        """
        Get all page images and metadata for a PDF by pdf_id.
        Returns dict with 'images', 'crawl_url', 'crawl_date', 'sub_domain' keys or error message.
        """
        if not pdf_id:
            return {"error": "Missing 'pdf_id' parameter"}, 400

        md = (self.metadata_index.search([pdf_id]) or {})
        first = (md.get(pdf_id) or [{}])[0]
        crawl_url = first.get("crawl_url", "")
        crawl_date = first.get("crawl_date", "")
        sub_domain = first.get("sub_domain", "")
        page_count = int(first.get("page_count", 0))

        image_dir = os.path.join(self.image_directory, pdf_id)
        images = [f"{image_dir}/{pdf_id}_{i}.jpeg" for i in range(page_count)]

        return {
            "images": images,
            "crawl_url": crawl_url,
            "crawl_date": crawl_date,
            "sub_domain": sub_domain,
        }

    def _get_total_pdfs_count(self):
        if (hasattr(self, '_total_pdfs_cache')):
            return self._total_pdfs_cache
        
        total_pdfs_path = self.stats_file
        
        if not total_pdfs_path or not os.path.exists(total_pdfs_path):
            return 0
        
        try:
            with open(total_pdfs_path, "r", encoding='utf-8') as f:
                content = f.read().strip()
                self._total_pdfs_cache =  int(content) 
                return self._total_pdfs_cache
                
        except Exception as e:
            print(f"Error reading total_pdfs.txt: {str(e)}")
            self._total_pdfs_cache = 0
            return 0

    def run(self, host="localhost", port=8080, debug=False):
        """Run the Flask server."""
        self.app.run(host=host, port=port, debug=debug, threaded=True)
