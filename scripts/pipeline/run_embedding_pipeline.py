import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass

import govscape as gs
from govscape.config import PDF_ARCHIVE_PDF_DIR, DataModel, digest_shards
from govscape.data_loader import RemoteDirectoryIterator, build_data_loader
from govscape.pipelining import run_pipelined
from govscape.utils import base_argument_parser, str2bool

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


@dataclass
class Batch:
    """One batch of PDFs on its way through the pipeline, in its own directory."""

    number: int
    shard: str
    pdf_iter: RemoteDirectoryIterator
    # The shard's position after this batch, saved once the batch is uploaded.
    checkpoint: dict
    pdf_files: list[str]
    data_model: DataModel


def batch_pdf_dir(batch_dm: DataModel) -> str:
    return os.path.join(batch_dm.data_dir, "PDFs")


def upload_batch(
    local_dm, remote_dm, data_loader, do_text_embedding, do_img_embedding, do_metadata
):
    if do_text_embedding or do_img_embedding or do_metadata:
        data_loader.upload_directory(
            local_dm.txt_directory, remote_dm.txt_directory, compress=True
        )
        data_loader.upload_directory(
            local_dm.image_directory, remote_dm.image_directory, compress=False
        )
        data_loader.upload_directory(
            local_dm.metadata_directory, remote_dm.metadata_directory, compress=True
        )
    if do_text_embedding:
        data_loader.upload_directory(
            local_dm.embedding_directory, remote_dm.embedding_directory, compress=True
        )
    if do_img_embedding:
        data_loader.upload_directory(
            local_dm.embedding_img_pg_directory,
            remote_dm.embedding_img_pg_directory,
            compress=True,
        )


# overall method that gets the files in batches and runs them through the
# pipeline
def main():
    # FIELDS TO SET --------------------------------------------------------
    parser = base_argument_parser(description="S3 EC2 Embedding Pipeline")
    parser.add_argument(
        "--text_model_type",
        type=str,
        help="The model type to use for text embedding",
        default="ST",
    )
    parser.add_argument(
        "--visual_model_type",
        type=str,
        help="The model type to use for visual embedding",
        default="CLIP",
    )
    parser.add_argument(
        "--num_servers",
        type=int,
        help="The number of servers to use for embedding",
        default=1,
    )
    parser.add_argument(
        "--server_id", type=int, help="The ID of the current server", default=0
    )
    parser.add_argument(
        "--shard_prefix_length",
        type=int,
        default=2,
        help="PDFs are split into shards by this many leading digest characters "
        "(32 ** length shards), which are dealt out across servers; each shard is "
        "checkpointed separately, so --num_servers can change between runs",
    )
    parser.add_argument(
        "--do_text_embedding",
        type=str2bool,
        help="Whether to do text embedding",
        default=True,
    )
    parser.add_argument(
        "--do_img_embedding",
        type=str2bool,
        help="Whether to do image embedding",
        default=True,
    )
    parser.add_argument(
        "--do_metadata_collection",
        type=str2bool,
        help="Whether to do metadata collection",
        default=True,
    )
    parser.add_argument(
        "--do_ocr",
        type=str2bool,
        help="Whether to run OCR extraction on page images",
        default=False,
    )
    parser.add_argument(
        "--ocr_type",
        type=str,
        help="OCR engine type (easyocr, olmocr, ocrmypdf)",
        default="easyocr",
    )
    parser.add_argument(
        "--ocr_languages",
        type=str,
        help="Comma-separated list of languages for OCR (e.g., 'en,fr')",
        default="en",
    )
    parser.add_argument(
        "--ocr_gpu",
        type=str2bool,
        help="Whether to use GPU for OCR processing",
        default=False,
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default=PDF_ARCHIVE_PDF_DIR,
        help="Directory containing PDFs within the PDF archive",
    )
    parser.add_argument(
        "--max_pdfs",
        type=int,
        default=None,
        help="Stop after this many PDFs (overrides --num_pages_to_process, which "
        "caps at 1000 PDFs per unit); by default all of this server's shards",
    )
    # No cap unless --num_pages_to_process or --max_pdfs is given.
    parser.set_defaults(num_pages_to_process=None)
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    work_dir = os.path.join(project_root, "data", "prod")
    # Anything left from an interrupted run is redone from the shard checkpoints.
    shutil.rmtree(work_dir, ignore_errors=True)
    local_dm = DataModel(work_dir)
    remote_dm = DataModel(args.remote_data_dir)
    local_perf_path = local_dm.performance_file_path(
        f"embedding_pipeline_{args.server_id}"
    )
    remote_perf_path = remote_dm.performance_file_path(
        f"embedding_pipeline_{args.server_id}"
    )

    os.makedirs(local_dm.data_dir, exist_ok=True)
    os.makedirs(local_dm.checkpoints_directory, exist_ok=True)
    os.makedirs(local_dm.performance_directory, exist_ok=True)

    # ---------------------------------------------------------------------------
    pipeline_times = {
        "list": 0,
        "download": 0,
        "pdf_to_txt_img_time": 0,
        "ocr_time": 0,
        "text_embed_time": 0,
        "img_embed_time": 0,
        "upload": 0,
        "pdfs_processed": 0,
    }
    # to keep track of the time it takes for each step in the pipeline

    data_loader = build_data_loader(
        args.backend,
        args.bucket_name,
        local_base_dir=args.local_base_dir,
        profile_name=args.profile,
        endpoint_url=args.endpoint_url,
    )
    pdf_data_loader = build_data_loader(
        args.backend,
        args.pdf_bucket_name,
        local_base_dir=args.local_base_dir,
        profile_name=args.pdf_profile,
        endpoint_url=args.endpoint_url,
    )
    _ocr_kwargs: dict = {}
    if args.do_ocr and args.ocr_type in ("easyocr", "olmocr"):
        _ocr_kwargs["languages"] = args.ocr_languages.split(",")
        _ocr_kwargs["gpu"] = args.ocr_gpu

    processor = gs.PDFProcessingPipeline(
        local_dm.data_dir,
        args.text_model_type,
        args.visual_model_type,
        ocr_type=args.ocr_type if args.do_ocr else None,
        **_ocr_kwargs,
    )

    overall_start_time = time.time()

    if args.max_pdfs is not None:
        max_files_to_process = args.max_pdfs
    elif args.num_pages_to_process is not None:
        max_files_to_process = args.num_pages_to_process * 1000
    else:
        max_files_to_process = math.inf
    shards = digest_shards(args.shard_prefix_length)[args.server_id :: args.num_servers]
    run_extraction = (
        args.do_text_embedding
        or args.do_img_embedding
        or args.do_metadata_collection
        or args.do_ocr
    )

    def download_batches():
        downloaded = 0
        number = 0
        for shard in shards:
            if downloaded >= max_files_to_process:
                return
            job_name = f"embedding_pipeline_shard_{shard}"
            # Threads, not processes: the other stages run in threads, and
            # forking a multi-threaded process can deadlock the children.
            pdf_iter = RemoteDirectoryIterator(
                pdf_data_loader,
                args.pdf_dir,
                remote_checkpoint_path=remote_dm.checkpoint_file_path(job_name),
                local_checkpoint_path=local_dm.checkpoint_file_path(job_name),
                local_dir=work_dir,
                use_multiprocessing=False,
                checkpoint_loader=data_loader,
                name_prefix=shard,
            )
            while downloaded < max_files_to_process:
                batch_dm = DataModel(os.path.join(work_dir, f"batch_{number}"))
                start = time.time()
                pdf_files = pdf_iter.download_batch(
                    max_keys=min(args.batch_size, max_files_to_process - downloaded),
                    filter_fn=lambda key: key.endswith(".pdf"),
                    local_dir=batch_pdf_dir(batch_dm),
                )
                pipeline_times["download"] += time.time() - start
                if not pdf_files:
                    break
                downloaded += len(pdf_files)
                logging.info(
                    f"Batch {number}: downloaded {len(pdf_files)} PDFs "
                    f"from shard {shard}"
                )
                yield Batch(
                    number,
                    shard,
                    pdf_iter,
                    pdf_iter.checkpoint_state(),
                    pdf_files,
                    batch_dm,
                )
                number += 1

    def extract(batch: Batch) -> Batch:
        if run_extraction:
            pipeline_times["pdf_to_txt_img_time"] += processor.extract(
                batch.pdf_files, batch.data_model, args.do_ocr
            )
            # Nothing reads the PDFs again, and several batches wait on disk for
            # the GPU.
            shutil.rmtree(batch_pdf_dir(batch.data_model))
        return batch

    def ocr(batch: Batch) -> Batch:
        pipeline_times["ocr_time"] += processor.ocr(batch.data_model)
        return batch

    def embed(batch: Batch) -> Batch:
        text_time, img_time = processor.embed(
            batch.data_model, args.do_text_embedding, args.do_img_embedding
        )
        pipeline_times["text_embed_time"] += text_time
        pipeline_times["img_embed_time"] += img_time
        return batch

    def upload(batch: Batch) -> None:
        start = time.time()
        upload_batch(
            batch.data_model,
            remote_dm,
            data_loader,
            args.do_text_embedding,
            args.do_img_embedding,
            args.do_metadata_collection,
        )
        pipeline_times["upload"] += time.time() - start
        # Batches are uploaded in order, so the checkpoint only ever moves past
        # batches whose results are on the remote.
        batch.pdf_iter.save_checkpoint(batch.checkpoint)
        shutil.rmtree(batch.data_model.data_dir)
        pipeline_times["pdfs_processed"] += len(batch.pdf_files)
        logging.info(f"Batch {batch.number} uploaded; pipeline times: {pipeline_times}")
        with open(local_perf_path, "w") as f:
            json.dump(pipeline_times, f, indent=2)
        data_loader.upload_file(local_perf_path, remote_perf_path)

    # While the GPU embeds one batch, the next is OCR'd, the one after that
    # extracted, the next downloaded, and the previous one uploaded.
    stages = [extract, ocr, embed, upload] if args.do_ocr else [extract, embed, upload]
    run_pipelined(download_batches(), stages)

    # After all batches are processed, clean up the directories
    if os.path.exists(local_dm.data_dir):
        shutil.rmtree(local_dm.data_dir)
        os.makedirs(local_dm.data_dir, exist_ok=True)

    overall_end_time = time.time()
    print("TOTAL TIME TO LOAD IS ", (overall_end_time - overall_start_time))
    print("TOTAL TIME list pdfs:", pipeline_times["list"])
    print("TOTAL TIME download pdfs:", pipeline_times["download"])
    print("TOTAL TIME pdf -> txt and img time:", pipeline_times["pdf_to_txt_img_time"])
    print("TOTAL TIME txt -> embed time:", pipeline_times["text_embed_time"])
    print("TOTAL TIME img -> embed time:", pipeline_times["img_embed_time"])
    print("TOTAL TIME uploading data:", pipeline_times["upload"])


if __name__ == "__main__":
    main()
