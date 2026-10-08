import math
import os
import uuid
from pathlib import Path

import pytest

import boto3
from botocore.client import BaseClient as S3Client
from botocore.config import Config

from govscape.data_loader import (
    DataLoader,
    LocalDataLoader,
    RemoteDirectoryIterator,
    S3DataLoader,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _create_s3_bucket(moto_endpoint: str) -> tuple[str, Config, S3Client]:
    bucket = f"test-bucket-{uuid.uuid4().hex[:12]}"
    config = Config(max_pool_connections=10)
    client = boto3.client("s3", endpoint_url=moto_endpoint, config=config)
    client.create_bucket(Bucket=bucket)
    return bucket, config, client


@pytest.fixture(params=["local", "s3", "s3_prefixed"])
def loader(request: pytest.FixtureRequest, tmp_path: Path) -> DataLoader:
    """Provide a DataLoader for each backend so a single test covers both.

    The ``local`` variant stores objects under ``tmp_path``. The ``s3`` variants
    talk to the session-scoped moto mock server (see ``tests/conftest.py``);
    each test gets a freshly created, uniquely named bucket for isolation.
    ``s3_prefixed`` stores everything under a key prefix within the bucket, as
    is done for the source.coop repositories.
    """
    if request.param == "local":
        return LocalDataLoader(base_dir=str(tmp_path / "data"))

    bucket, config, client = _create_s3_bucket(request.getfixturevalue("moto_server"))
    if request.param == "s3_prefixed":
        bucket = f"{bucket}/org/repo/"
    return S3DataLoader(bucket_name=bucket, config=config, s3_client=client)


def _remote_basenames(loader: DataLoader, prefix: str) -> list[str]:
    """Return the sorted basenames of objects stored under ``prefix``.

    Uses the backend-agnostic ``list_objects`` interface so the same assertion
    works for both the local and S3 loaders.
    """
    keys = loader.list_objects(prefix, max_keys=100000).keys
    return sorted(os.path.basename(key) for key in keys)


@pytest.mark.parametrize("use_multiprocessing", [False, True])
def test_data_loader_continuation_token(
    loader: DataLoader, tmp_path: Path, use_multiprocessing: bool
) -> None:
    checkpoint_path = "checkpoint.json"
    local_checkpoint_path = tmp_path / "local_checkpoint.json"

    download_dir = tmp_path / "download"

    # Create 5 files under a prefix.
    prefix = "files"
    for i in range(5):
        loader.upload_bytes(b"x", f"{prefix}/file_{i}.txt")

    remote_iter = RemoteDirectoryIterator(
        loader,
        prefix,
        checkpoint_path,
        str(local_checkpoint_path),
        local_dir=str(download_dir),
        use_multiprocessing=use_multiprocessing,
    )

    # First page
    result1 = remote_iter.download_batch(max_keys=2)
    assert len(result1) == 2
    remote_iter.save_checkpoint()

    # Second page
    result2 = remote_iter.download_batch(max_keys=2)
    assert len(result2) == 2
    remote_iter.save_checkpoint()

    remote_iter.close()

    # New iter should resume from checkpoint
    remote_iter2 = RemoteDirectoryIterator(
        loader,
        prefix,
        checkpoint_path,
        str(local_checkpoint_path),
        local_dir=str(download_dir),
        use_multiprocessing=use_multiprocessing,
    )
    result3 = remote_iter2.download_batch(max_keys=2)
    assert len(result3) == 1

    # No more files to download
    result4 = remote_iter2.download_batch(max_keys=2)
    assert len(result4) == 0

    remote_iter2.close()


def test_upload_directory_compressed(loader: DataLoader, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    remote_prefix = "uploaded"

    num_files = 2500
    chunk_size = 1000
    for i in range(num_files):
        _touch(source_dir / f"subdir_{i // 100}" / f"file_{i}.txt")

    loader.upload_directory(
        str(source_dir), remote_prefix, compress=True, chunk_size=chunk_size
    )

    uploaded_names = _remote_basenames(loader, remote_prefix)

    # Build the expected chunk names using the same hash logic as the
    # implementation: collect all files sorted, split into chunks, and hash
    # each chunk's file-path tuple.
    all_files: list[str] = [
        os.path.join(root, filename)
        for root, _, files in os.walk(str(source_dir))
        for filename in files
    ]
    all_files.sort()

    expected_names: list[str] = []
    for idx in range(0, len(all_files), chunk_size):
        chunk_files = all_files[idx : idx + chunk_size]
        expected_names.append(f"chunk_{hash(tuple(chunk_files))}.tar.gz")
    expected_names.sort()

    expected_chunks = math.ceil(num_files / chunk_size)
    assert len(uploaded_names) == expected_chunks
    assert all(name.endswith(".tar.gz") for name in uploaded_names)
    assert uploaded_names == expected_names


def test_download_file_decompress(loader: DataLoader, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    download_dir = tmp_path / "download"
    remote_prefix = "uploaded"

    # Create files in subdirectories, upload compressed, then download with decompress.
    num_files = 3
    subdirs = ["alpha", "beta"]
    for subdir in subdirs:
        for i in range(num_files):
            _touch(source_dir / subdir / f"file_{i}.txt")

    loader.upload_directory(
        str(source_dir), remote_prefix, compress=True, chunk_size=1000
    )

    # There should be exactly one .tar.gz chunk.
    tar_keys = loader.list_objects(remote_prefix, max_keys=100000).keys
    assert len(tar_keys) == 1
    remote_tar_path = tar_keys[0]
    tar_name = os.path.basename(remote_tar_path)
    local_tar_path = str(download_dir / tar_name)

    loader.download_file(remote_tar_path, local_tar_path, decompress=True)

    # The .tar.gz should have been deleted after extraction.
    assert not os.path.exists(local_tar_path)

    # The subdirectories should have been preserved after extraction.
    extracted_subdirs = sorted(d.name for d in download_dir.iterdir() if d.is_dir())
    assert extracted_subdirs == sorted(subdirs)

    # Each subdirectory should contain the original files.
    for subdir in subdirs:
        extracted = sorted(f.name for f in (download_dir / subdir).iterdir())
        expected = sorted(f"file_{i}.txt" for i in range(num_files))
        assert extracted == expected


def test_download_directory(loader: DataLoader, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    download_dir = tmp_path / "download"
    remote_prefix = "uploaded"

    _touch(source_dir / "subdir_a" / "file_1.txt")
    _touch(source_dir / "subdir_b" / "file_2.txt")
    _touch(source_dir / "file_3.txt")

    loader.upload_directory(str(source_dir), remote_prefix)
    loader.download_directory(remote_prefix, str(download_dir))

    assert (download_dir / "subdir_a" / "file_1.txt").exists()
    assert (download_dir / "subdir_b" / "file_2.txt").exists()
    assert (download_dir / "file_3.txt").exists()


@pytest.mark.parametrize("use_multiprocessing", [False, True])
def test_remote_directory_iterator_compressed(
    loader: DataLoader, tmp_path: Path, use_multiprocessing: bool
) -> None:
    """Round-trip test: upload with compression, iterate with RemoteDirectoryIterator.

    Verifies that RemoteDirectoryIterator transparently decompresses .tar.gz
    chunks, returns the correct extracted file paths, respects filter_fn,
    handles multiple chunks via batching, and works with checkpointing.
    """
    download_dir = tmp_path / "download"
    source_dir = tmp_path / "source"
    remote_prefix = "compressed_files"
    checkpoint_path = "checkpoints/checkpoint.json"
    local_checkpoint_path = str(tmp_path / "local_checkpoint.json")

    # Create 15 .npy files and 5 .txt files across subdirectories so we can
    # verify filter_fn works on the extracted contents.
    npy_files_created: list[str] = []
    for i in range(15):
        subdir = f"doc_{i // 5}"
        name = f"embedding_{i}.npy"
        _touch(source_dir / subdir / name)
        npy_files_created.append(os.path.join(subdir, name))
    for i in range(5):
        subdir = f"doc_{i // 5}"
        _touch(source_dir / subdir / f"notes_{i}.txt")
    npy_files_created.sort()

    # Upload with small chunk_size to force multiple .tar.gz archives.
    loader.upload_directory(str(source_dir), remote_prefix, compress=True, chunk_size=8)

    # Verify multiple chunks were created.
    tar_names = _remote_basenames(loader, remote_prefix)
    assert all(name.endswith(".tar.gz") for name in tar_names)
    assert len(tar_names) >= 2, "Expected multiple compressed chunks"

    # --- Batch 1: download first batch with a filter for .npy files only ---
    remote_iter = RemoteDirectoryIterator(
        loader,
        remote_prefix,
        checkpoint_path,
        local_checkpoint_path,
        local_dir=str(download_dir),
        use_multiprocessing=use_multiprocessing,
    )

    batch1 = remote_iter.download_batch(
        max_keys=4,
        filter_fn=lambda key: key.endswith(".npy"),
    )

    # Only .npy files should be returned.
    assert len(batch1) == 15
    assert all(p.endswith(".npy") for p in batch1)
    # No .tar.gz files should remain in the download directory.
    remaining_tar = [
        f for f in Path(str(download_dir)).rglob("*") if f.name.endswith(".tar.gz")
    ]
    assert len(remaining_tar) == 0, "tar.gz files should be cleaned up"

    # All .npy files from the source should have been extracted.
    extracted_npy = sorted(os.path.relpath(p, str(download_dir)) for p in batch1)
    assert extracted_npy == npy_files_created

    # Subdirectory structure should be preserved.
    extracted_dirs = sorted(
        d.name for d in Path(str(download_dir)).iterdir() if d.is_dir()
    )
    assert "doc_0" in extracted_dirs
    assert "doc_1" in extracted_dirs
    assert "doc_2" in extracted_dirs

    # --- Verify checkpointing: a new iterator should have nothing left ---
    remote_iter.save_checkpoint()
    remote_iter.close()

    remote_iter2 = RemoteDirectoryIterator(
        loader,
        remote_prefix,
        checkpoint_path,
        local_checkpoint_path,
        local_dir=str(download_dir),
        use_multiprocessing=use_multiprocessing,
    )
    batch2 = remote_iter2.download_batch(max_keys=4)
    # The first iterator consumed all pages, so the second should be empty.
    assert len(batch2) == 0
    remote_iter2.close()


def test_s3_prefixed_keys(moto_server: str, tmp_path: Path) -> None:
    bucket, config, client = _create_s3_bucket(moto_server)
    loader = S3DataLoader(
        bucket_name=f"{bucket}/org/repo", config=config, s3_client=client
    )

    loader.upload_bytes(b"x", "files/a.txt")
    _touch(tmp_path / "source" / "b.txt")
    loader.upload_directory(str(tmp_path / "source"), "dir")

    raw_keys = sorted(
        obj["Key"] for obj in client.list_objects_v2(Bucket=bucket)["Contents"]
    )
    assert raw_keys == ["org/repo/dir/b.txt", "org/repo/files/a.txt"]
    assert loader.list_objects("files").keys == ["files/a.txt"]
    assert loader.to_uri("files/a.txt") == (
        f"{moto_server}/{bucket}/org/repo/files/a.txt"
    )


@pytest.mark.parametrize("use_multiprocessing", [False, True])
def test_remote_directory_iterator_checkpoint_loader(
    moto_server: str, tmp_path: Path, use_multiprocessing: bool
) -> None:
    src_bucket, config, client = _create_s3_bucket(moto_server)
    ckpt_bucket, _, _ = _create_s3_bucket(moto_server)
    src_loader = S3DataLoader(
        bucket_name=f"{src_bucket}/archive", config=config, s3_client=client
    )
    ckpt_loader = S3DataLoader(
        bucket_name=f"{ckpt_bucket}/derived", config=config, s3_client=client
    )
    for i in range(3):
        src_loader.upload_bytes(b"x", f"pdfs/file_{i}.pdf")

    remote_iter = RemoteDirectoryIterator(
        src_loader,
        "pdfs/",
        "checkpoints/ckpt.json",
        str(tmp_path / "ckpt.json"),
        local_dir=str(tmp_path / "download"),
        use_multiprocessing=use_multiprocessing,
        checkpoint_loader=ckpt_loader,
    )
    downloaded = remote_iter.download_batch(max_keys=10)
    remote_iter.save_checkpoint()
    remote_iter.close()

    assert sorted(os.path.basename(p) for p in downloaded) == [
        f"file_{i}.pdf" for i in range(3)
    ]
    assert ckpt_loader.exists("checkpoints/ckpt.json")
    assert not src_loader.exists("checkpoints/ckpt.json")


def test_list_objects_treats_prefix_as_directory(loader: DataLoader) -> None:
    loader.upload_bytes(b"x", "embeddings/a.npy")
    loader.upload_bytes(b"x", "embeddings_img_pg/b.npy")

    assert loader.list_objects("embeddings").keys == ["embeddings/a.npy"]
    assert loader.list_objects("embeddings/").keys == ["embeddings/a.npy"]
