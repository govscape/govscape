import os
import shutil
from collections.abc import Iterator

import pytest

dirs_to_remove = [
    "tests/test_data/old_Test_Data/small/text",
    "tests/test_data/old_Test_Data/small/embeddings",
    "tests/test_data/old_Test_Data/small/images",
]


@pytest.fixture(scope="session", autouse=True)
def aws_credentials() -> Iterator[None]:
    """Provide dummy AWS credentials so boto3 and s5cmd can sign requests.

    These point at nothing real; all S3 traffic is served by the moto mock
    server (see ``moto_server``).
    """
    fake_env = {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }
    previous = {key: os.environ.get(key) for key in fake_env}
    os.environ.update(fake_env)
    yield
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def moto_server(aws_credentials: None) -> Iterator[str]:
    """Start an in-process mock S3 HTTP server for the test session.

    Running moto in *server* mode (rather than the in-process ``mock_aws``
    decorator) is required because ``S3DataLoader`` reaches S3 through the
    ``s5cmd`` subprocess and through forked multiprocessing workers, neither of
    which an in-process monkeypatch can intercept. Pointing ``S3_ENDPOINT_URL``
    (read by s5cmd) and ``AWS_ENDPOINT_URL`` (read by botocore, inherited by
    forked workers) at the server redirects all of them to the mock.
    """
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(port=0)
    server.start()
    _, port = server.get_host_and_port()
    endpoint = f"http://127.0.0.1:{port}"

    previous = {
        key: os.environ.get(key) for key in ("AWS_ENDPOINT_URL", "S3_ENDPOINT_URL")
    }
    os.environ["AWS_ENDPOINT_URL"] = endpoint
    os.environ["S3_ENDPOINT_URL"] = endpoint
    try:
        yield endpoint
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        server.stop()


@pytest.fixture(scope="session", autouse=True)
def cleanup_directories():
    # Pre-Test Setup (nothing to be done)
    yield
    # Post-Test Cleanup
    print("Cleaning up test directories...")
    for directory in dirs_to_remove:
        shutil.rmtree(directory, ignore_errors=True)
