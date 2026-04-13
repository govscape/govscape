# These classes should contain all the configuration information necessary for
# starting the server and serving queries, respectively.
import os


class DataModel:
    """Defines the subdirectory layout within a data directory."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.txt_directory = os.path.join(data_dir, "txt")
        self.embedding_directory = os.path.join(data_dir, "embeddings")
        self.embedding_img_pg_directory = os.path.join(data_dir, "embeddings_img_pg")
        self.index_directory = os.path.join(data_dir, "index")
        self.index_img_pg_directory = os.path.join(data_dir, "index_img_pg")
        self.index_keyword_directory = os.path.join(data_dir, "index_keyword")
        self.index_metadata_directory = os.path.join(data_dir, "index_metadata")
        self.image_directory = os.path.join(data_dir, "img")
        self.metadata_directory = os.path.join(data_dir, "metadata")
        self.checkpoints_directory = os.path.join(data_dir, "checkpoints")
        self.performance_directory = os.path.join(data_dir, "performance")
        self.stats_file = os.path.join(data_dir, "total_pdfs.txt")
        self.blacklist_file = os.path.join(data_dir, "blacklist.txt")


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
