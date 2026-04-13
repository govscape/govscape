import logging
import os
import shlex
import subprocess
import sys

from govscape.config import DataModel
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

    local_dm = DataModel(LOCAL_DATA_DIR)
    remote_dm = DataModel(REMOTE_DATA_DIR)

    REMOTE_CHECKPOINT_PATH = os.path.join(
        remote_dm.checkpoints_directory, "checkpoint_server.json"
    )
    LOCAL_CHECKPOINT_PATH = os.path.join(
        local_dm.checkpoints_directory, "checkpoint_server.json"
    )
    data_loader = build_data_loader(
        args.backend, BUCKET_NAME, local_base_dir=LOCAL_MOCK_DIR
    )
    for remote_dir, local_dir in [
        (remote_dm.index_keyword_directory, local_dm.index_keyword_directory),
        (remote_dm.index_directory, local_dm.index_directory),
        (remote_dm.index_img_pg_directory, local_dm.index_img_pg_directory),
        (remote_dm.index_metadata_directory, local_dm.index_metadata_directory),
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

    remote_blacklist = remote_dm.blacklist_file
    local_blacklist = local_dm.blacklist_file
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
