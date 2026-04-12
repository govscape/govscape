import logging
import os
import shlex
import subprocess
import sys

from govscape.data_loader import RemoteDirectoryIterator, build_data_loader

from .start_api_server import _get_arg_parser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def download_indices(args):
    # This function can be used to download the index files from S3 before starting
    # the server. It can be called separately if you want to ensure the indices are
    # downloaded before starting the server.
    BUCKET_NAME = args.bucket_name
    LOCAL_MOCK_DIR = args.local_base_dir
    REMOTE_DATA_DIR = args.remote_data_directory
    LOCAL_DATA_DIR = args.local_data_directory
    REMOTE_KEYWORD_INDEX_DIR = os.path.join(REMOTE_DATA_DIR, "index_keyword")
    LOCAL_KEYWORD_INDEX_DIR = os.path.join(LOCAL_DATA_DIR, "index_keyword")
    REMOTE_TEXT_EMBEDDING_INDEX_DIR = os.path.join(REMOTE_DATA_DIR, "index")
    LOCAL_TEXT_EMBEDDING_INDEX_DIR = os.path.join(LOCAL_DATA_DIR, "index")
    REMOTE_VISUAL_EMBEDDING_INDEX_DIR = os.path.join(REMOTE_DATA_DIR, "index_img_pg")
    LOCAL_VISUAL_EMBEDDING_INDEX_DIR = os.path.join(LOCAL_DATA_DIR, "index_img_pg")
    REMOTE_METADATA_INDEX_DIR = os.path.join(REMOTE_DATA_DIR, "index_metadata")
    LOCAL_METADATA_INDEX_DIR = os.path.join(LOCAL_DATA_DIR, "index_metadata")
    REMOTE_CHECKPOINT_PATH = os.path.join(
        REMOTE_DATA_DIR, "checkpoints", "checkpoint_server.json"
    )
    LOCAL_CHECKPOINT_PATH = os.path.join(
        LOCAL_DATA_DIR, "checkpoints", "checkpoint_server.json"
    )
    data_loader = build_data_loader(
        args.backend, BUCKET_NAME, local_base_dir=LOCAL_MOCK_DIR
    )
    for remote_dir, local_dir in [
        (REMOTE_KEYWORD_INDEX_DIR, LOCAL_KEYWORD_INDEX_DIR),
        (REMOTE_TEXT_EMBEDDING_INDEX_DIR, LOCAL_TEXT_EMBEDDING_INDEX_DIR),
        (REMOTE_VISUAL_EMBEDDING_INDEX_DIR, LOCAL_VISUAL_EMBEDDING_INDEX_DIR),
        (REMOTE_METADATA_INDEX_DIR, LOCAL_METADATA_INDEX_DIR),
    ]:
        remote_iter = RemoteDirectoryIterator(
            data_loader,
            remote_dir,
            REMOTE_CHECKPOINT_PATH,
            LOCAL_CHECKPOINT_PATH,
            local_dir,
        )
        finished = False
        while not finished:
            downloaded_files = remote_iter.download_batch()
            if len(downloaded_files) == 0:
                finished = True
                print(f"Finished downloading indices from {remote_dir} to {local_dir}")

    remote_blacklist = os.path.join(REMOTE_DATA_DIR, "blacklist.txt")
    local_blacklist = os.path.join(LOCAL_DATA_DIR, "blacklist.txt")
    try:
        data_loader.download_file(remote_blacklist, local_blacklist)
        print(f"Downloaded blacklist: {remote_blacklist} -> {local_blacklist}")
    except Exception as e:
        print(f"No blacklist file to download ({e}); proceeding without one")


def main():
    if "--" in sys.argv:
        sep_index = sys.argv.index("--")
        app_argv = sys.argv[1:sep_index]
        gunicorn_argv = sys.argv[sep_index + 1 :]
    else:
        app_argv = sys.argv[1:]
        gunicorn_argv = [
            "gunicorn",
            "-c",
            "gunicorn.conf.py",
            "scripts.serving.start_api_server:create_app()",
        ]

    parser = _get_arg_parser()
    args = parser.parse_args(app_argv)
    download_indices(args)

    # Set APP_ARGS so create_app() can parse them without interfering with gunicorn argv
    os.environ["APP_ARGS"] = " ".join(shlex.quote(a) for a in app_argv)

    print("Running:", " ".join(gunicorn_argv))
    sys.exit(subprocess.call(gunicorn_argv))


if __name__ == "__main__":
    main()
