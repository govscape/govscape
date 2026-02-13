# AI modified: 2025-02-13 525524a525e5ade4b8c714baa8be2c27c0027393
import math
import os
from pathlib import Path

from govscape.data_loader import LocalDataLoader, RemoteDirectoryIterator


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def test_local_data_loader_continuation_token(tmp_path: Path) -> None:
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
    all_files: list[str] = []
    for root, _, files in os.walk(str(source_dir)):
        for filename in files:
            all_files.append(os.path.join(root, filename))
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
