import os
from pathlib import Path

from govscape.data_loader import LocalDataLoader


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def test_local_data_loader_continuation_token(tmp_path: Path) -> None:
    base_dir = tmp_path / "data"
    checkpoint_path = tmp_path / "checkpoint.json"

    # Create 5 files under a prefix directory
    prefix = "files"
    for i in range(5):
        _touch(base_dir / prefix / f"file_{i}.txt")

    loader = LocalDataLoader(str(base_dir), checkpoint_path=str(checkpoint_path))

    # First page
    result1 = loader.list_objects(prefix, max_keys=2)
    assert len(result1.keys) == 2
    assert result1.is_truncated is True
    loader.save_checkpoint()

    # Second page
    result2 = loader.list_objects(prefix, max_keys=2)
    assert len(result2.keys) == 2
    assert result2.is_truncated is True
    loader.save_checkpoint()

    # New loader should resume from checkpoint
    loader2 = LocalDataLoader(str(base_dir), checkpoint_path=str(checkpoint_path))
    result3 = loader2.list_objects(prefix, max_keys=2)
    assert len(result3.keys) == 1
    assert result3.is_truncated is False
