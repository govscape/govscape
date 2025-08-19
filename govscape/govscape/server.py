# This file defines the logic for serving requests to the user.
from flask import Flask, send_from_directory
from flask_cors import CORS
from .config import ServerConfig
import numpy as np
import os
import struct
import json
import fcntl
import time
import math
from .api import init_api
from .filter import Filter
from .indexing import DiskANNIndex, FAISSIndex, WhooshIndex, SQLiteMetadataIndex

# basic pipeline developed:
# 1. accept a query until EOF detected
# 2. run an embedding model on the query
# 3. return a list of files that are most similar to the query - utilize FAISS to do this
class Server:
    # obtain all the setup information from configuration
    def __init__(self, config: ServerConfig):
        self.config = config

        # Directories
        self.pdf_directory = config.pdf_directory
        self.metadata_directory = config.metadata_directory
        self.embedding_directory = config.embedding_directory
        self.embedding_img_pg_directory = config.embedding_img_pg_directory
        self.index_directory = config.index_directory
        self.index_img_pg_directory = config.index_img_pg_directory
        self.index_keyword_directory = config.index_keyword_directory
        self.index_metadata_directory = config.index_metadata_directory
        self.image_directory = config.image_directory
        self.stats_file = config.stats_file
        self.index_type = config.index_type
        self.k = config.k

        # Index configuration
        self.index_config = config.index_config

        # Model Params
        self.text_model = config.text_model
        self.text_d = config.text_d

        self.visual_model = config.visual_model
        self.visual_d = config.visual_d

        if self.index_type == 'Disk':
            self.text_index = DiskANNIndex(self.embedding_directory, self.index_directory)
            self.text_index.load_index()
            self.visual_index = DiskANNIndex(self.embedding_directory, self.index_img_pg_directory)
            self.visual_index.load_index()
        elif self.index_type == 'Memory':
            self.text_index = FAISSIndex(self.index_directory)
            self.text_index.load_index()
            self.visual_index = FAISSIndex(self.index_img_pg_directory)
            self.visual_index.load_index()
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        self.keyword_index = WhooshIndex(self.index_keyword_directory)
        self.keyword_index.load_index()

        self.metadata_index = SQLiteMetadataIndex(self.index_metadata_directory)
        self.metadata_index.load_index()

        self.filt = Filter(config)

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
        CORS(self.app, origins=["http://3.20.135.189", "http://localhost:5173"], supports_credentials=True)

        @self.app.route("/")
        def serve_index():
            print("Serving index.html")
            return self.app.send_static_file("index.html")

        @self.app.route("/images/<path:filename>")
        def serve_image(filename):
            image_dir = os.path.abspath(self.image_directory)
            return send_from_directory(image_dir, filename)

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
        while len(search_results) < results_needed_for_page:
            # Search for the k closest arrays
            D, pdf_names, pdf_pages = index.search(query_embedding, current_k)

            D, pdf_names, pdf_pages = self.deduplicate_responses(D, pdf_names, pdf_pages)
            
            pdf_metadata = self.metadata_index.search(pdf_names, filters)
                
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
                        "crawl_url" : metadata.get("url", ""),
                        "crawl_date": metadata.get("crawl_date", ""),
                        "sub_domain": metadata.get("sub_domain", ""),
                    })

            if current_k > min(100000, index.total_entries()): 
                break # TODO: If we have to expand beyond 100k, we should simply do the filtering first

            if len(search_results) >= results_needed_for_page:
                break # If we have enough results for our target page, we can stop.
            
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
        """Get all page images for a PDF by pdf_id. Returns dict with 'images' key or error message."""
        if not pdf_id:
            return {"error": "Missing 'pdf_id' parameter"}, 400

        metadata_path = os.path.join(self.metadata_directory, pdf_id, "metadata.json")
        
        if not os.path.exists(metadata_path):
            return {"error": "Metadata not found"}, 404

        with open(metadata_path, "r") as f:
            meta = json.load(f)
        num_pages = meta.get("num_pages")
        if not num_pages:
            return {"error": "Page number not found"}, 404

        try:
            num_pages = int(num_pages)
        except Exception:
            raise Exception(f"Invalid page number in metadata: {metadata_path}")

        image_dir = os.path.join(self.image_directory, pdf_id)
        images = [f"{image_dir}/{pdf_id}_{i}.jpeg" for i in range(num_pages)]
        return {"images": images}

    def _get_total_pdfs_count(self):
        # TODO: use Redis to cache the total number of PDFs
        current_time = time.time()
        cache_duration = 3600  # 1 hour in seconds
        
        if (hasattr(self, '_total_pdfs_cache') and 
            hasattr(self, '_total_pdfs_cache_time') and
            current_time - self._total_pdfs_cache_time < cache_duration):
            return self._total_pdfs_cache
        
        total_pdfs_path = self.stats_file
        
        if not total_pdfs_path or not os.path.exists(total_pdfs_path):
            self._total_pdfs_cache = 0
            self._total_pdfs_cache_time = current_time
            return 0
        
        try:
            with open(total_pdfs_path, "r", encoding='utf-8') as f:
                locked = False
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    locked = True
                except (OSError, IOError):
                    pass
                
                content = f.read().strip()
                if not content:
                    val = 0
                else:
                    try:
                        val = int(content)
                    except ValueError:
                        val = 0
                
                if locked:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
                
                self._total_pdfs_cache = val
                self._total_pdfs_cache_time = current_time
                
                return val
                
        except Exception as e:
            print(f"Error reading total_pdfs.txt: {str(e)}")
            self._total_pdfs_cache = 0
            self._total_pdfs_cache_time = current_time
            return 0

    def serve(self):
        # keep this function to maintain compatibility with scripts/start_server.py
        print("Welcome to End-Of-Term PDF Search Server")

        print("Searching against " + str(self.index.total_entries()) + " embeddings\n")
        try:
            while True:
                query = input("Search: ")
                if query == "":
                    continue

                result = self.search(query)
                print(json.dumps(result, indent=4))

        except EOFError:
            print("\nThank you for using!")

    def run(self, host="localhost", port=8080, debug=False):
        """Run the Flask server."""
        self.app.run(host=host, port=port, debug=debug, threaded=True)
