# These classes should contain all the configuration information necessary for
# starting the server and serving queries, respectively.
import os

# Remote storage on source.coop. Raw PDFs and the CDX live in the PDF archive;
# all derived data (txt, img, embeddings, indices, ...) live in the derivative
# repository. See DATA_MODEL.md for the layout of each.
# Data is accessed through the source.coop S3 proxy, where the account name
# ("govscape") acts as the bucket and each repository is a key prefix.
SOURCE_COOP_ENDPOINT = "https://data.source.coop"
SOURCE_COOP_PREFIX = "govscape"
PDF_ARCHIVE_BUCKET = f"{SOURCE_COOP_PREFIX}/eota-pdf-archive/"
DERIVATIVE_BUCKET = f"{SOURCE_COOP_PREFIX}/eota-derivative-data/"
PDF_ARCHIVE_PDF_DIR = "pdfs/"
PDF_ARCHIVE_CDX_KEY = "cdx/complete_cdx.parquet"


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

    # Per-PDF subdirectory helpers

    def txt_pdf_directory(self, digest: str) -> str:
        return os.path.join(self.txt_directory, digest)

    def img_pdf_directory(self, digest: str) -> str:
        return os.path.join(self.image_directory, digest)

    def embedding_pdf_directory(self, digest: str) -> str:
        return os.path.join(self.embedding_directory, digest)

    def embedding_img_pg_pdf_directory(self, digest: str) -> str:
        return os.path.join(self.embedding_img_pg_directory, digest)

    def metadata_pdf_directory(self, digest: str) -> str:
        return os.path.join(self.metadata_directory, digest)

    # Per-page file helpers

    def txt_page_path(self, digest: str, pg_no: int) -> str:
        return os.path.join(self.txt_directory, digest, f"{digest}_{pg_no}.txt")

    def img_page_path(self, digest: str, pg_no: int) -> str:
        return os.path.join(self.image_directory, digest, f"{digest}_{pg_no}.jpeg")

    def embedding_page_path(self, digest: str, pg_no: int) -> str:
        return os.path.join(self.embedding_directory, digest, f"{digest}_{pg_no}.npy")

    def embedding_img_pg_page_path(self, digest: str, pg_no: int) -> str:
        return os.path.join(
            self.embedding_img_pg_directory, digest, f"{digest}_{pg_no}.npy"
        )

    def metadata_file_path(self, digest: str) -> str:
        return os.path.join(self.metadata_directory, digest, "metadata.json")

    def checkpoint_file_path(self, checkpoint_prefix: str) -> str:
        return os.path.join(
            self.checkpoints_directory, f"{checkpoint_prefix}_checkpoint.json"
        )

    def performance_file_path(self, performance_prefix: str) -> str:
        return os.path.join(
            self.performance_directory, f"{performance_prefix}_performance.json"
        )


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
