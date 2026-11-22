import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest


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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def moto_server(aws_credentials: None) -> Iterator[str]:
    """Start a mock S3 HTTP server for the test session.

    Running moto in *server* mode (rather than the in-process ``mock_aws``
    decorator) is required because ``S3DataLoader`` reaches S3 through the
    ``s5cmd`` subprocess and through forked multiprocessing workers, neither of
    which an in-process monkeypatch can intercept. Pointing ``S3_ENDPOINT_URL``
    (read by s5cmd) and ``AWS_ENDPOINT_URL`` (read by botocore, inherited by
    forked workers) at the server redirects all of them to the mock.

    The server runs in its own process: in-process (``ThreadedMotoServer``) its
    request threads are alive whenever the tests fork pool workers, and a worker
    forked while one of them holds a lock can deadlock, hanging the test.
    """
    port = _free_port()
    endpoint = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "moto.server", "-H", "127.0.0.1", "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while True:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            if server.poll() is not None or time.monotonic() > deadline:
                server.kill()
                raise RuntimeError("moto server failed to start") from None
            time.sleep(0.1)

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
        server.terminate()
        server.wait(timeout=10)
