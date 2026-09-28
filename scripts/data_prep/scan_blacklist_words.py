"""Scan extracted PDF text pages for blacklisted words in parallel.

Downloads {digest}_{pg_no}.txt pages from the remote txt/ directory in
batches, groups pages by digest, and scans each digest's pages across a
pool of worker processes. Digests containing a blacklisted word are
appended to the PDF-digest blacklist.txt (see DATA_MODEL.md) so they are
hidden from search results.
"""

import json
import logging
import os
import shutil
import time
from multiprocessing import Pool, cpu_count

from govscape.config import DataModel
from govscape.data_loader import DataLoader, RemoteDirectoryIterator, build_data_loader
from govscape.utils import (
    base_argument_parser,
    contains_blacklisted_word,
    load_word_blacklist,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


_word_blacklist: set[str] | None = None


def _init_worker(word_blacklist: set[str]) -> None:
    """Pool initializer: load the blacklist once per worker instead of per task."""
    global _word_blacklist
    _word_blacklist = word_blacklist


def _scan_digest(digest_and_paths: tuple[str, list[str]]) -> tuple[str, bool]:
    """Worker task: return `digest` and whether any of its pages contain a blacklisted
    word.
    """
    digest, txt_paths = digest_and_paths
    assert _word_blacklist is not None
    for txt_path in txt_paths:
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()
        if contains_blacklisted_word(text, _word_blacklist):
            return digest, True
    return digest, False


def _tag_metadata_contains_blacklisted_word(
    data_loader: DataLoader,
    local_dm: DataModel,
    remote_dm: DataModel,
    digest: str,
    contains_word: bool,
) -> None:
    """Download a digest's metadata.json, stamp contains_blacklisted_word, upload it
    back.
    """
    remote_path = remote_dm.metadata_file_path(digest)
    local_path = local_dm.metadata_file_path(digest)
    try:
        data_loader.download_file(remote_path, local_path)
    except Exception as e:
        print(f"No metadata.json found for {digest} ({e}); skipping")
        return

    with open(local_path, encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["contains_blacklisted_word"] = contains_word
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    data_loader.upload_file(local_path, remote_path)


def _load_pdf_blacklist(path: str) -> set[str]:
    """Load the PDF-digest blacklist.txt (see DATA_MODEL.md).

    Entries are case-sensitive.
    """
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


if __name__ == "__main__":
    parser = base_argument_parser(
        description="Scan extracted text pages for blacklisted words"
    )
    parser.set_defaults(batch_size=100000)
    parser.add_argument(
        "--word_blacklist_path",
        type=str,
        required=True,
        help="Local path to a newline-delimited word blacklist file",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of parallel worker processes used to scan documents",
    )
    args = parser.parse_args()

    NUM_PAGES_TO_PROCESS = args.num_pages_to_process
    BATCH_SIZE = args.batch_size
    BUCKET_NAME = args.bucket_name
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    LOCAL_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "prod")
    REMOTE_DATA_DIR = args.remote_data_dir

    local_dm = DataModel(LOCAL_DATA_DIR)
    remote_dm = DataModel(REMOTE_DATA_DIR)
    REMOTE_CHECKPOINT_PATH = os.path.join(
        remote_dm.checkpoints_directory, "checkpoint_blacklist_scan.json"
    )
    LOCAL_CHECKPOINT_PATH = local_dm.checkpoint_file_path("blacklist_scan")

    os.makedirs(local_dm.txt_directory, exist_ok=True)
    os.makedirs(local_dm.checkpoints_directory, exist_ok=True)

    word_blacklist = load_word_blacklist(args.word_blacklist_path)
    if not word_blacklist:
        raise ValueError(f"No blacklist words loaded from {args.word_blacklist_path}")

    data_loader = build_data_loader(
        args.backend,
        BUCKET_NAME,
        local_base_dir=args.local_base_dir,
    )

    remote_iter = RemoteDirectoryIterator(
        data_loader,
        remote_dm.txt_directory,
        remote_checkpoint_path=REMOTE_CHECKPOINT_PATH,
        local_checkpoint_path=LOCAL_CHECKPOINT_PATH,
        local_dir=local_dm.txt_directory,
    )

    try:
        data_loader.download_file(remote_dm.blacklist_file, local_dm.blacklist_file)
    except Exception as e:
        print(f"No existing PDF blacklist to download ({e}); starting fresh")
    pdf_blacklist = _load_pdf_blacklist(local_dm.blacklist_file)

    def scan_txt_files(txt_files: list[str]) -> None:
        # Group downloaded pages by digest so each worker task scans one document.
        pages_by_digest: dict[str, list[str]] = {}
        for txt_file in txt_files:
            digest = os.path.basename(os.path.dirname(txt_file))
            pages_by_digest.setdefault(digest, []).append(txt_file)

        tasks = list(pages_by_digest.items())
        num_workers = min(args.num_workers, len(tasks))
        with Pool(
            processes=num_workers, initializer=_init_worker, initargs=(word_blacklist,)
        ) as pool:
            results = pool.map(_scan_digest, tasks)

        for digest, contains_word in results:
            _tag_metadata_contains_blacklisted_word(
                data_loader, local_dm, remote_dm, digest, contains_word
            )

        new_matches = {
            digest
            for digest, contains_word in results
            if contains_word and digest not in pdf_blacklist
        }
        if new_matches:
            pdf_blacklist.update(new_matches)
            print(f"Blacklisting {len(new_matches)} digest(s): {sorted(new_matches)}")
            with open(local_dm.blacklist_file, "a", encoding="utf-8") as f:
                for digest in sorted(new_matches):
                    f.write(digest + "\n")
            data_loader.upload_file(local_dm.blacklist_file, remote_dm.blacklist_file)

    def batched_scan(batch_size: int) -> None:
        files_processed = 0
        overall_start_time = time.time()
        while files_processed < NUM_PAGES_TO_PROCESS:
            batch_limit = min(batch_size, NUM_PAGES_TO_PROCESS - files_processed)
            local_paths = remote_iter.download_batch(
                max_keys=batch_limit,
                filter_fn=lambda key: key.endswith(".txt"),
            )
            if not local_paths:
                break

            scan_txt_files(local_paths)
            remote_iter.save_checkpoint()
            files_processed += len(local_paths)

            if os.path.exists(local_dm.txt_directory):
                shutil.rmtree(local_dm.txt_directory)
                os.makedirs(local_dm.txt_directory, exist_ok=True)

        print(
            f"Scanned {files_processed} page(s) in "
            f"{time.time() - overall_start_time:.1f}s"
        )

    try:
        batched_scan(BATCH_SIZE)
    finally:
        remote_iter.close()
