from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Self

import boto3
from botocore.client import BaseClient as S3Client
from botocore.config import Config
from botocore.exceptions import ClientError


@dataclass
class ListResult:
    keys: list[str]
    is_truncated: bool
    continuation_token: str | None


class DataLoader(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> ListResult:
        raise NotImplementedError

    @abstractmethod
    def download_file(
        self, remote_path: str, local_path: str, decompress: bool = False
    ) -> list[str]:
        raise NotImplementedError

    @staticmethod
    def _decompress_tar_gz(tar_path: str) -> list[str]:
        """Extract a .tar.gz archive into its parent directory and remove it.

        Extraction happens in a private temporary directory first and the
        results are moved to the final destination so that concurrent calls
        operating on the same parent directory do not race with each other.

        Returns the list of extracted file paths (excludes directories).
        """
        extract_dir = os.path.dirname(tar_path)
        extracted_files: list[str] = []

        def move_extracted_file(tmp_file: str, rel_path: str) -> str:
            dest = os.path.join(extract_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(tmp_file, dest)
            return dest

        with tempfile.TemporaryDirectory() as tmp_dir:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=tmp_dir)

            extracted_paths: list[tuple[str, str]] = []
            for root, _, files in os.walk(tmp_dir):
                for filename in files:
                    tmp_file = os.path.join(root, filename)
                    rel_path = os.path.relpath(tmp_file, tmp_dir)
                    extracted_paths.append((tmp_file, rel_path))

            if extracted_paths:
                max_workers = min(32, len(extracted_paths))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    extracted_files.extend(
                        executor.map(
                            lambda item: move_extracted_file(item[0], item[1]),
                            extracted_paths,
                        )
                    )
        os.remove(tar_path)
        return extracted_files

    @abstractmethod
    def download_directory(self, remote_prefix: str, local_dir: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_bytes(self, data: bytes, remote_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_directory(
        self,
        local_dir: str,
        remote_prefix: str,
        compress: bool = False,
        chunk_size: int = 1000,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def _create_tar_chunk(args: tuple[list[str], str, str]) -> str:
        """Create one compressed tar chunk from a list of files."""
        chunk_files, local_dir, tar_path = args
        with tarfile.open(tar_path, "w:gz") as tar:
            for file_path in chunk_files:
                arcname = os.path.relpath(file_path, local_dir)
                tar.add(file_path, arcname=arcname)
        return tar_path

    def _upload_directory_compressed(
        self, local_dir: str, remote_prefix: str, chunk_size: int = 1000
    ) -> None:
        """Upload a directory as a series of compressed .tar.gz chunks.

        Each chunk archive contains at most *chunk_size* files.  Archives are
        named using a hash of their contents for uniqueness across concurrent
        uploaders, e.g. ``chunk_<hash>.tar.gz``, and uploaded under
        *remote_prefix*.
        """
        all_files: list[str] = [
            os.path.join(root, filename)
            for root, _, files in os.walk(local_dir)
            for filename in files
        ]
        all_files.sort()

        # If there are no files to upload, do nothing.
        if not all_files:
            return

        def build_chunk_task(
            chunk_idx: int, tmp_dir: str
        ) -> tuple[list[str], str, str]:
            chunk_files = all_files[chunk_idx : chunk_idx + chunk_size]
            chunk_hash = hash(tuple(chunk_files))
            chunk_name = f"chunk_{chunk_hash}.tar.gz"
            tar_path = os.path.join(tmp_dir, chunk_name)
            return (chunk_files, local_dir, tar_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_indices = list(range(0, len(all_files), chunk_size))
            tasks = [
                build_chunk_task(chunk_idx, tmp_dir) for chunk_idx in chunk_indices
            ]
            if tasks:
                workers = min(os.cpu_count() or 1, len(tasks))
                with Pool(processes=workers) as pool:
                    pool.map(self._create_tar_chunk, tasks)
            self.upload_directory(tmp_dir, remote_prefix, compress=False)

    @abstractmethod
    def copy_object(self, source_path: str, dest_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_object(self, remote_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_uri(self, remote_path: str) -> str:
        raise NotImplementedError


class S3DataLoader(DataLoader):
    def __init__(
        self,
        bucket_name: str,
        config: Config,
        s3_client: S3Client | None = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.s3: S3Client = s3_client or boto3.client("s3", config=config)

    def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> ListResult:
        kwargs = {
            "Bucket": self.bucket_name,
            "Prefix": prefix,
            "MaxKeys": max_keys,
        }
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        remaining = max_keys
        is_truncated = False
        keys = []
        while remaining > 0:
            kwargs["MaxKeys"] = min(10000, remaining)
            result = self.s3.list_objects_v2(**kwargs)
            contents = result.get("Contents", [])
            keys.extend([obj["Key"] for obj in contents])
            continuation_token = result.get("NextContinuationToken")
            kwargs["ContinuationToken"] = continuation_token
            remaining = max_keys - len(keys)
            is_truncated = result.get("IsTruncated", False)
            if not continuation_token:
                break
        return ListResult(
            keys=keys,
            is_truncated=is_truncated,
            continuation_token=continuation_token,
        )

    def download_file(
        self, remote_path: str, local_path: str, decompress: bool = False
    ) -> list[str]:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket_name, remote_path, local_path)
        if decompress and local_path.endswith(".tar.gz"):
            return self._decompress_tar_gz(local_path)
        return [local_path]

    def download_directory(self, remote_prefix: str, local_dir: str) -> None:
        os.makedirs(local_dir, exist_ok=True)
        normalized_prefix = remote_prefix.rstrip("/") + "/"
        normalized_local_dir = local_dir.rstrip("/") + "/"
        subprocess.run(
            [
                "poetry",
                "run",
                "s5cmd",
                "--log",
                "error",
                "sync",
                f"s3://{self.bucket_name}/{normalized_prefix}",
                normalized_local_dir,
            ],
            check=True,
        )

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self.s3.upload_file(local_path, self.bucket_name, remote_path)

    def upload_bytes(self, data: bytes, remote_path: str) -> None:
        self.s3.put_object(Bucket=self.bucket_name, Key=remote_path, Body=data)

    def upload_directory(
        self,
        local_dir: str,
        remote_prefix: str,
        compress: bool = False,
        chunk_size: int = 1000,
    ) -> None:
        if compress:
            self._upload_directory_compressed(local_dir, remote_prefix, chunk_size)
            return
        normalized_local_dir = local_dir.rstrip("/") + "/"
        normalized_prefix = remote_prefix.rstrip("/") + "/"
        subprocess.run(
            [
                "poetry",
                "run",
                "s5cmd",
                "--log",
                "error",
                "cp",
                normalized_local_dir,
                f"s3://{self.bucket_name}/{normalized_prefix}",
            ],
            check=True,
        )

    def copy_object(self, source_path: str, dest_path: str) -> None:
        self.s3.copy_object(
            Bucket=self.bucket_name,
            CopySource=f"/{self.bucket_name}/{source_path}",
            Key=dest_path,
        )

    def delete_object(self, remote_path: str) -> None:
        self.s3.delete_object(Bucket=self.bucket_name, Key=remote_path)

    def exists(self, remote_path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=remote_path)
            return True
        except ClientError:
            return False

    def to_uri(self, remote_path: str) -> str:
        return f"https://{self.bucket_name}.s3.amazonaws.com/{remote_path}"


class LocalDataLoader(DataLoader):
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir

    def _resolve(self, remote_path: str) -> str:
        return os.path.join(self.base_dir, remote_path)

    def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> ListResult:
        start_index = int(continuation_token) if continuation_token else 0
        base_path = self._resolve(prefix)
        keys: list[str] = []
        if os.path.exists(base_path):
            for root, _, files in os.walk(base_path):
                for filename in files:
                    abs_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(abs_path, self.base_dir)
                    keys.append(rel_path.replace("\\", "/"))
        keys.sort()
        slice_keys = keys[start_index : start_index + max_keys]
        next_index = start_index + max_keys
        continuation_token = str(next_index) if next_index < len(keys) else None
        return ListResult(
            keys=slice_keys,
            is_truncated=continuation_token is not None,
            continuation_token=continuation_token,
        )

    def download_file(
        self, remote_path: str, local_path: str, decompress: bool = False
    ) -> list[str]:
        source_path = self._resolve(remote_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(source_path, local_path)
        if decompress and local_path.endswith(".tar.gz"):
            return self._decompress_tar_gz(local_path)
        return [local_path]

    def download_directory(self, remote_prefix: str, local_dir: str) -> None:
        source = self._resolve(remote_prefix)
        if not os.path.exists(source):
            return
        os.makedirs(local_dir, exist_ok=True)
        for root, _, files in os.walk(source):
            for filename in files:
                src_path = os.path.join(root, filename)
                rel_path = os.path.relpath(src_path, source)
                dest_path = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        dest_path = self._resolve(remote_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(local_path, dest_path)

    def upload_bytes(self, data: bytes, remote_path: str) -> None:
        dest_path = self._resolve(remote_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(data)

    def upload_directory(
        self,
        local_dir: str,
        remote_prefix: str,
        compress: bool = False,
        chunk_size: int = 1000,
    ) -> None:
        if compress:
            self._upload_directory_compressed(local_dir, remote_prefix, chunk_size)
            return
        for root, _, files in os.walk(local_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, local_dir)
                dest_path = os.path.join(self._resolve(remote_prefix), rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(local_path, dest_path)

    def copy_object(self, source_path: str, dest_path: str) -> None:
        source_abs = self._resolve(source_path)
        dest_abs = self._resolve(dest_path)
        os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
        shutil.copy2(source_abs, dest_abs)

    def delete_object(self, remote_path: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            os.remove(self._resolve(remote_path))

    def exists(self, remote_path: str) -> bool:
        return os.path.exists(self._resolve(remote_path))

    def to_uri(self, remote_path: str) -> str:
        return f"file://{self._resolve(remote_path)}"


def build_data_loader(
    backend: str,
    bucket_name: str | None = None,
    local_base_dir: str | None = None,
    config: Config | None = None,
) -> DataLoader:
    if config is None:
        config = Config(max_pool_connections=60)
    if backend == "local":
        if not local_base_dir:
            raise ValueError("local_base_dir is required for local backend")
        return LocalDataLoader(base_dir=local_base_dir)
    if backend == "s3":
        if not bucket_name:
            raise ValueError("bucket_name is required for s3 backend")
        return S3DataLoader(
            bucket_name=bucket_name,
            config=config,
        )
    raise ValueError(f"Unsupported backend: {backend}")


# -- Multiprocessing helpers (module-level for picklability) --

_mp_loader: DataLoader | None = None


def _init_mp_worker(loader_type: str, loader_kwargs: dict) -> None:
    """Initializer for multiprocessing pool workers.

    Reconstructs the DataLoader in each worker process to avoid pickling
    boto3 clients.
    """
    global _mp_loader
    if loader_type == "s3":
        _mp_loader = S3DataLoader(
            bucket_name=loader_kwargs["bucket_name"],
            config=Config(
                max_pool_connections=loader_kwargs.get("max_pool_connections", 60)
            ),
        )
    elif loader_type == "local":
        _mp_loader = LocalDataLoader(base_dir=loader_kwargs["base_dir"])
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")


def _mp_download_file(args: tuple[str, str, bool]) -> list[str]:
    """Worker function that downloads a single file using the process-local loader."""
    remote_path, local_path, decompress = args
    assert _mp_loader is not None
    return _mp_loader.download_file(remote_path, local_path, decompress)


class RemoteDirectoryIterator:
    def __init__(
        self,
        data_loader: DataLoader,
        prefix: str,
        remote_checkpoint_path: str,
        local_checkpoint_path: str,
        local_dir: str,
        use_multiprocessing: bool = True,
    ) -> None:
        self.data_loader = data_loader
        self.prefix = prefix
        self.remote_checkpoint_path = remote_checkpoint_path
        self.local_dir = local_dir
        self.local_checkpoint_path = local_checkpoint_path
        self.use_multiprocessing = use_multiprocessing
        self.finished: bool = False
        self._continuation_token: str | None = None
        self._mp_pool: Any | None = None
        self._load_checkpoint_from_remote()

    def _get_mp_pool(self, num_workers: int):
        """Return the cached multiprocessing pool, creating it on first use."""
        if self._mp_pool is None:
            loader_type, loader_kwargs = self._get_loader_spec()
            self._mp_pool = Pool(
                processes=num_workers,
                initializer=_init_mp_worker,
                initargs=(loader_type, loader_kwargs),
            )
        return self._mp_pool

    def close(self) -> None:
        """Shut down the cached multiprocessing pool, if any."""
        if self._mp_pool is not None:
            self._mp_pool.terminate()
            self._mp_pool.join()
            self._mp_pool = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _load_checkpoint_from_remote(self) -> None:
        if self.data_loader.exists(self.remote_checkpoint_path):
            self.data_loader.download_file(
                self.remote_checkpoint_path, self.local_checkpoint_path
            )
            with open(self.local_checkpoint_path) as f:
                checkpoint_data = json.load(f)
                self._continuation_token = checkpoint_data.get("continuation_token")
                self.finished = checkpoint_data.get("finished", False)
        else:
            self._continuation_token = None

    def save_checkpoint(self) -> None:
        checkpoint_dir = os.path.dirname(self.local_checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(self.local_checkpoint_path, "w") as f:
            json.dump(
                {
                    "continuation_token": self._continuation_token,
                    "finished": self.finished,
                },
                f,
            )
        self.data_loader.upload_file(
            self.local_checkpoint_path, self.remote_checkpoint_path
        )

    @staticmethod
    def _build_local_path(prefix: str, remote_path: str, local_dir: str) -> str:
        rel_path = remote_path
        if prefix and remote_path.startswith(prefix):
            rel_path = remote_path[len(prefix) :]
        rel_path = rel_path.lstrip("/")
        if not rel_path:
            rel_path = os.path.basename(remote_path)
        return os.path.join(local_dir, rel_path)

    def download_batch(
        self,
        max_keys: int = 1000,
        filter_fn: Callable[[str], bool] | None = None,
        num_workers: int | None = None,
    ) -> list[str]:
        if self.finished:
            return []
        downloaded_paths: list[str] = []
        remaining = max_keys
        while remaining > 0:
            result = self.data_loader.list_objects(
                self.prefix,
                max_keys=min(100000, remaining),
                continuation_token=self._continuation_token,
            )
            self._continuation_token = result.continuation_token
            keys = result.keys
            # Separate compressed archives from regular files so .tar.gz
            # archives can be decompressed and their contents returned.
            tar_keys = [k for k in keys if k.endswith(".tar.gz")]
            regular_keys = [k for k in keys if not k.endswith(".tar.gz")]
            if filter_fn:
                regular_keys = [k for k in regular_keys if filter_fn(k)]

            download_fn = (
                self._download_files_multiprocess
                if self.use_multiprocessing
                else self._download_files_multithreaded
            )

            if regular_keys:
                pairs = [
                    (key, self._build_local_path(self.prefix, key, self.local_dir))
                    for key in regular_keys
                ]
                downloaded_paths.extend(download_fn(pairs, num_workers=num_workers))

            if tar_keys:
                tar_pairs = [
                    (key, self._build_local_path(self.prefix, key, self.local_dir))
                    for key in tar_keys
                ]
                extracted = download_fn(
                    tar_pairs, num_workers=num_workers, decompress=True
                )
                downloaded_paths.extend(
                    [
                        fpath
                        for fpath in extracted
                        if filter_fn is None or filter_fn(fpath)
                    ]
                )

            if not result.is_truncated:
                self.finished = True
                break

            if regular_keys or tar_keys:
                remaining = max_keys - len(regular_keys) - len(tar_keys)
                if remaining <= 0:
                    break
        return downloaded_paths

    def _download_files_multithreaded(
        self,
        pairs: list[tuple[str, str]],
        num_workers: int | None = None,
        decompress: bool = False,
    ) -> list[str]:
        if num_workers is None:
            num_workers = min(10, (os.cpu_count() or 1) * 5)

        max_workers = min(num_workers, len(pairs))
        remote_paths = [pair[0] for pair in pairs]
        local_paths = [pair[1] for pair in pairs]
        decompress_flags = [decompress] * len(pairs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                self.data_loader.download_file,
                remote_paths,
                local_paths,
                decompress_flags,
            )
            return [path for file_list in results for path in file_list]

    def _get_loader_spec(self) -> tuple[str, dict]:
        """Return a picklable (type, kwargs) pair describing the data loader."""
        if isinstance(self.data_loader, S3DataLoader):
            return ("s3", {"bucket_name": self.data_loader.bucket_name})
        if isinstance(self.data_loader, LocalDataLoader):
            return ("local", {"base_dir": self.data_loader.base_dir})
        raise TypeError(f"Cannot serialize loader: {type(self.data_loader)}")

    def _download_files_multiprocess(
        self,
        pairs: list[tuple[str, str]],
        num_workers: int | None = None,
        decompress: bool = False,
    ) -> list[str]:
        if num_workers is None:
            num_workers = (os.cpu_count() or 1) * 2

        max_workers = min(num_workers, len(pairs))
        pool = self._get_mp_pool(max_workers)
        tasks = [(remote, local, decompress) for remote, local in pairs]
        results = pool.map(_mp_download_file, tasks)
        return [path for file_list in results for path in file_list]
