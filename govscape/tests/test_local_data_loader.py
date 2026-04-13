import math
import os
from pathlib import Path

import pytest

from govscape.data_loader import LocalDataLoader, RemoteDirectoryIterator


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


@pytest.mark.parametrize("use_multiprocessing", [False, True])
def test_local_data_loader_continuation_token(
    tmp_path: Path, use_multiprocessing: bool
) -> None:
    base_dir = tmp_path / "data"
    checkpoint_path = tmp_path / "checkpoint.json"
    local_checkpoint_path = tmp_path / "local_checkpoint.json"

    download_dir = tmp_path / "download"

    # Create 5 files under a prefix directory
    prefix = "files"
    for i in range(5):
        _touch(base_dir / prefix / f"file_{i}.txt")

    loader = LocalDataLoader(base_dir=str(base_dir))
    remote_iter = RemoteDirectoryIterator(
        loader,
        prefix,
        str(checkpoint_path),
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

    # New iter should resume from checkpoint
    remote_iter2 = RemoteDirectoryIterator(
        loader,
        prefix,
        str(checkpoint_path),
        str(local_checkpoint_path),
        local_dir=str(download_dir),
        use_multiprocessing=use_multiprocessing,
    )
    result3 = remote_iter2.download_batch(max_keys=2)
    assert len(result3) == 1

    # No more files to download
    result4 = remote_iter2.download_batch(max_keys=2)
    assert len(result4) == 0


def test_upload_directory_compressed(tmp_path: Path) -> None:
    base_dir = tmp_path / "data"
    source_dir = tmp_path / "source"
    remote_prefix = "uploaded"

    num_files = 2500
    chunk_size = 1000
    for i in range(num_files):
        _touch(source_dir / f"subdir_{i // 100}" / f"file_{i}.txt")

    loader = LocalDataLoader(base_dir=str(base_dir))
    loader.upload_directory(
        str(source_dir), remote_prefix, compress=True, chunk_size=chunk_size
    )

    dest_dir = base_dir / remote_prefix
    uploaded_files = sorted(dest_dir.iterdir())

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
    assert len(uploaded_files) == expected_chunks
    assert all(f.name.endswith(".tar.gz") for f in uploaded_files)
    assert [f.name for f in uploaded_files] == expected_names


def test_download_file_decompress(tmp_path: Path) -> None:
    base_dir = tmp_path / "data"
    source_dir = tmp_path / "source"
    download_dir = tmp_path / "download"
    remote_prefix = "uploaded"

    # Create files in subdirectories, upload compressed, then download with decompress.
    num_files = 3
    subdirs = ["alpha", "beta"]
    for subdir in subdirs:
        for i in range(num_files):
            _touch(source_dir / subdir / f"file_{i}.txt")

    loader = LocalDataLoader(base_dir=str(base_dir))
    loader.upload_directory(
        str(source_dir), remote_prefix, compress=True, chunk_size=1000
    )

    # There should be exactly one .tar.gz chunk.
    dest_dir = base_dir / remote_prefix
    tar_files = list(dest_dir.iterdir())
    assert len(tar_files) == 1
    tar_name = tar_files[0].name

    remote_tar_path = f"{remote_prefix}/{tar_name}"
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


def test_download_directory(tmp_path: Path) -> None:
    base_dir = tmp_path / "data"
    source_dir = tmp_path / "source"
    download_dir = tmp_path / "download"
    remote_prefix = "uploaded"

    _touch(source_dir / "subdir_a" / "file_1.txt")
    _touch(source_dir / "subdir_b" / "file_2.txt")
    _touch(source_dir / "file_3.txt")

    loader = LocalDataLoader(base_dir=str(base_dir))
    loader.upload_directory(str(source_dir), remote_prefix)
    loader.download_directory(remote_prefix, str(download_dir))

    assert (download_dir / "subdir_a" / "file_1.txt").exists()
    assert (download_dir / "subdir_b" / "file_2.txt").exists()
    assert (download_dir / "file_3.txt").exists()


@pytest.mark.parametrize("use_multiprocessing", [False, True])
def test_remote_directory_iterator_compressed(
    tmp_path: Path, use_multiprocessing: bool
) -> None:
    """Round-trip test: upload with compression, iterate with RemoteDirectoryIterator.

    Verifies that RemoteDirectoryIterator transparently decompresses .tar.gz
    chunks, returns the correct extracted file paths, respects filter_fn,
    handles multiple chunks via batching, and works with checkpointing.
    """
    base_dir = tmp_path / "data"
    download_dir = tmp_path / "download"
    source_dir = tmp_path / "source"
    remote_prefix = "compressed_files"
    checkpoint_path = os.path.join(str(base_dir), "checkpoints", "checkpoint.json")
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

    loader = LocalDataLoader(base_dir=str(base_dir))

    # Upload with small chunk_size to force multiple .tar.gz archives.
    loader.upload_directory(str(source_dir), remote_prefix, compress=True, chunk_size=8)

    # Verify multiple chunks were created.
    dest_dir = base_dir / remote_prefix
    tar_files = [f for f in dest_dir.iterdir() if f.name.endswith(".tar.gz")]
    assert len(tar_files) >= 2, "Expected multiple compressed chunks"

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
