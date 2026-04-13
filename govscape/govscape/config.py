# These classes should contain all the configuration information necessary for
# starting the server and serving queries, respectively.
import numpy as np


class IndexConfig:
    def __init__(self, data_dir, vector_index_type, keyword_index_type):
        self.embedding_directory = data_dir + "/embeddings"
        self.embedding_img_pg_directory = data_dir + "/embeddings_img_pg"
        self.index_directory = data_dir + "/index"
        self.index_img_pg_directory = data_dir + "/index_img_pg"
        self.index_keyword_directory = data_dir + "/index_keyword"
        self.index_metadata_directory = data_dir + "/index_metadata"
        self.image_directory = data_dir + "/img"
        self.metadata_directory = data_dir + "/metadata"
        self.stats_file = data_dir + "/total_pdfs.txt"
        if vector_index_type not in ["Memory", "Disk"]:
            raise ValueError("vector_index_type must be either 'Memory' or 'Disk'")
        self.vector_index_type = vector_index_type
        self.keyword_index_type = keyword_index_type
        if keyword_index_type not in ["LanceDB", "SQLite", "Whoosh", "Lucene"]:
            raise ValueError(
                "keyword_index_type must be 'LanceDB', 'SQLite', 'Whoosh', or 'Lucene'"
            )
        self.dtype = np.float32


class ServerConfig:
    def __init__(
        self,
        index_config: IndexConfig,
        text_model,
        visual_model,
        k=3,
        prefilter_document_threshold=1000,
    ):
        self.index_config = index_config
        self.embedding_directory = index_config.embedding_directory
        self.embedding_img_pg_directory = index_config.embedding_img_pg_directory
        self.index_directory = index_config.index_directory
        self.index_img_pg_directory = index_config.index_img_pg_directory
        self.index_keyword_directory = index_config.index_keyword_directory
        self.index_metadata_directory = index_config.index_metadata_directory
        self.image_directory = index_config.image_directory
        self.metadata_directory = index_config.metadata_directory
        self.stats_file = index_config.stats_file
        self.text_model = text_model
        self.visual_model = visual_model
        self.vector_index_type = index_config.vector_index_type
        self.keyword_index_type = index_config.keyword_index_type

        # define k for top-k
        self.k = k
        self.prefilter_document_threshold = prefilter_document_threshold

        # define embedding size
        self.text_d = self.text_model.d
        self.visual_d = self.visual_model.d
