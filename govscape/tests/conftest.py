import shutil

import pytest

dirs_to_remove = [
    "tests/test_data/small/text",
    "tests/test_data/small/embeddings",
    "tests/test_data/small/images",
]


@pytest.fixture(scope="session", autouse=True)
def cleanup_directories():
    # Pre-Test Setup (nothing to be done)
    yield
    # Post-Test Cleanup
    print("Cleaning up test directories...")
    for directory in dirs_to_remove:
        shutil.rmtree(directory, ignore_errors=True)
