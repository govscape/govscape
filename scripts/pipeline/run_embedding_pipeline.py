import json
import logging
import os
import shutil
import time

from govscape.config import DataModel
from govscape.data_loader import RemoteDirectoryIterator, build_data_loader
from govscape.utils import base_argument_parser, str2bool

import govscape as gs

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# processing the pdfs: running through embedding pipeline and uploading to s3
def process_pdfs(
    pdf_files,
    processor,
    do_text_embedding,
    do_img_embedding,
    do_metadata_collection,
    pipeline_times,
    data_loader,
    local_dm,
    remote_dm,
):
    print("Do_Text_embedding: ", do_text_embedding)
    print("Do_Img_embedding: ", do_img_embedding)
    print("Do_Metadata_collection: ", do_metadata_collection)

    # PROCESS PDFS HERE
    pdf_to_txt_img_time, text_embed_time, img_embed_time = processor.process_pdfs(
        pdf_files, do_text_embedding, do_img_embedding, do_metadata_collection
    )
    pipeline_times["pdf_to_txt_img_time"] += pdf_to_txt_img_time
    pipeline_times["text_embed_time"] += text_embed_time
    pipeline_times["img_embed_time"] += img_embed_time

    time1 = time.time()
    # UPLOADING EMBEDDINGS, TXTS, IMAGES TO S3 HERE
    if do_text_embedding or do_img_embedding or do_metadata_collection:
        data_loader.upload_directory(
            local_dm.txt_directory, remote_dm.txt_directory, compress=True
        )
        print("finished uploading txt")

        data_loader.upload_directory(
            local_dm.image_directory, remote_dm.image_directory, compress=False
        )
        print("finished uploading img")

        data_loader.upload_directory(
            local_dm.metadata_directory, remote_dm.metadata_directory, compress=True
        )
        print("finished uploading metadata")

    if do_text_embedding:
        data_loader.upload_directory(
            local_dm.embedding_directory,
            remote_dm.embedding_directory,
            compress=True,
        )
        print("finished uploading embeddings")

    if do_img_embedding:
        data_loader.upload_directory(
            local_dm.embedding_img_pg_directory,
            remote_dm.embedding_img_pg_directory,
            compress=True,
        )
        print("finished uploading embed img pg")
    time2 = time.time()

    pipeline_times["upload"] += time2 - time1
    pipeline_times["pdfs_processed"] += len(pdf_files)

    print("finished uploading current batch")
    print("pipeline times: ", pipeline_times)


def cleanup(local_dm, local_pdf_dir):
    if os.path.exists(local_dm.data_dir):
        if os.path.exists(local_dm.txt_directory):
            shutil.rmtree(local_dm.txt_directory)
        if os.path.exists(local_dm.image_directory):
            shutil.rmtree(local_dm.image_directory)
        if os.path.exists(local_dm.embedding_directory):
            shutil.rmtree(local_dm.embedding_directory)
        if os.path.exists(local_dm.embedding_img_pg_directory):
            shutil.rmtree(local_dm.embedding_img_pg_directory)
        if os.path.exists(local_dm.metadata_directory):
            shutil.rmtree(local_dm.metadata_directory)
        os.makedirs(local_dm.data_dir, exist_ok=True)

    if os.path.exists(local_pdf_dir):
        shutil.rmtree(local_pdf_dir)
        os.makedirs(local_pdf_dir, exist_ok=True)


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
    parser.add_argument("--pdf_dir", type=str, help="Directory containing PDFs")
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    local_dm = DataModel(os.path.join(project_root, "data", "prod"))
    remote_dm = DataModel(args.remote_data_dir)
    local_pdf_dir = os.path.join(local_dm.data_dir, "PDFs")
    local_checkpoint_path = local_dm.checkpoint_file_path(
        f"embedding_pipeline_{args.server_id}"
    )
    remote_checkpoint_path = remote_dm.checkpoint_file_path(
        f"embedding_pipeline_{args.server_id}"
    )
    local_perf_path = local_dm.performance_file_path(
        f"embedding_pipeline_{args.server_id}"
    )
    remote_perf_path = remote_dm.performance_file_path(
        f"embedding_pipeline_{args.server_id}"
    )

    os.makedirs(local_dm.data_dir, exist_ok=True)
    os.makedirs(local_pdf_dir, exist_ok=True)
    os.makedirs(local_dm.checkpoints_directory, exist_ok=True)
    os.makedirs(local_dm.performance_directory, exist_ok=True)

    # ---------------------------------------------------------------------------
    pipeline_times = {
        "list": 0,
        "download": 0,
        "pdf_to_txt_img_time": 0,
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
    )
    remote_pdf_iter = RemoteDirectoryIterator(
        data_loader,
        args.pdf_dir,
        remote_checkpoint_path=remote_checkpoint_path,
        local_checkpoint_path=local_checkpoint_path,
        local_dir=local_pdf_dir,
    )

    processor = gs.PDFProcessingPipeline(
        local_dm.data_dir, args.text_model_type, args.visual_model_type
    )

    overall_start_time = time.time()

    max_files_to_process = args.num_pages_to_process * 1000
    files_processed = 0
    while files_processed < max_files_to_process:
        print("-" * 93)
        print("FILES PROCESSED: ", files_processed)
        print("-" * 93)
        time_download = time.time()
        os.makedirs(local_pdf_dir, exist_ok=True)
        batch_limit = min(args.batch_size, max_files_to_process - files_processed)
        local_batch = remote_pdf_iter.download_batch(
            max_keys=batch_limit,
            filter_fn=lambda key: (
                key.endswith(".pdf")
                and (hash(key) % args.num_servers) == args.server_id
            ),
        )
        pipeline_times["download"] += time.time() - time_download
        if not local_batch:
            break
        print("len(local_batch) = ", len(local_batch))

        process_pdfs(
            local_batch,
            processor,
            args.do_text_embedding,
            args.do_img_embedding,
            args.do_metadata_collection,
            pipeline_times,
            data_loader,
            local_dm,
            remote_dm,
        )

        remote_pdf_iter.save_checkpoint()
        files_processed += len(local_batch)

        # Delete local data to free up space for the next batch
        cleanup(local_dm, local_pdf_dir)

        # Write pipeline_times to a JSON file
        with open(local_perf_path, "w") as f:
            json.dump(pipeline_times, f, indent=2)

        # Upload the performance JSON to S3
        data_loader.upload_file(local_perf_path, remote_perf_path)

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
