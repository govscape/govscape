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
