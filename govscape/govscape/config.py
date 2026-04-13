# These classes should contain all the configuration information necessary for
# starting the server and serving queries, respectively.


class DataModel:
    """Defines the subdirectory layout within a data directory."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.txt_directory = data_dir + "/txt"
        self.embedding_directory = data_dir + "/embeddings"
        self.embedding_img_pg_directory = data_dir + "/embeddings_img_pg"
        self.index_directory = data_dir + "/index"
        self.index_img_pg_directory = data_dir + "/index_img_pg"
        self.index_keyword_directory = data_dir + "/index_keyword"
        self.index_metadata_directory = data_dir + "/index_metadata"
        self.image_directory = data_dir + "/img"
        self.metadata_directory = data_dir + "/metadata"
        self.checkpoints_directory = data_dir + "/checkpoints"
        self.performance_directory = data_dir + "/performance"
        self.stats_file = data_dir + "/total_pdfs.txt"
        self.blacklist_file = data_dir + "/blacklist.txt"


class ServerConfig:
    def __init__(
        self,
        data_dir,
        text_model,
        visual_model,
        vector_index_type,
        keyword_index_type,
        k=3,
        max_crawl_instances=500,
    ):
        self.data_model = DataModel(data_dir)
        self.embedding_directory = self.data_model.embedding_directory
        self.embedding_img_pg_directory = self.data_model.embedding_img_pg_directory
        self.index_directory = self.data_model.index_directory
        self.index_img_pg_directory = self.data_model.index_img_pg_directory
        self.index_keyword_directory = self.data_model.index_keyword_directory
        self.index_metadata_directory = self.data_model.index_metadata_directory
        self.image_directory = self.data_model.image_directory
        self.metadata_directory = self.data_model.metadata_directory
        self.stats_file = self.data_model.stats_file
        self.blacklist_file = self.data_model.blacklist_file
        self.text_model = text_model
        self.visual_model = visual_model

        if vector_index_type not in ["Memory", "Disk"]:
            raise ValueError("vector_index_type must be either 'Memory' or 'Disk'")
        self.vector_index_type = vector_index_type

        if keyword_index_type not in ["LanceDB", "SQLite", "Whoosh", "Lucene"]:
            raise ValueError(
                "keyword_index_type must be 'LanceDB', 'SQLite', 'Whoosh', or 'Lucene'"
            )
        self.keyword_index_type = keyword_index_type
        self.max_crawl_instances = max_crawl_instances

        # define k for top-k
        self.k = k

        # define embedding size
        self.text_d = self.text_model.d
        self.visual_d = self.visual_model.d
