from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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
    def download_file(self, remote_path: str, local_path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_bytes(self, data: bytes, remote_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_directory(self, local_dir: str, remote_prefix: str) -> None:
        raise NotImplementedError

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
        result = self.s3.list_objects_v2(**kwargs)
        contents = result.get("Contents", [])
        keys = [obj["Key"] for obj in contents]
        self._continuation_token = result.get("NextContinuationToken")
        return ListResult(
            keys=keys,
            is_truncated=result.get("IsTruncated", False),
            continuation_token=self._continuation_token,
        )

    def download_file(self, remote_path: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket_name, remote_path, local_path)
        return local_path

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self.s3.upload_file(local_path, self.bucket_name, remote_path)

    def upload_bytes(self, data: bytes, remote_path: str) -> None:
        self.s3.put_object(Bucket=self.bucket_name, Key=remote_path, Body=data)

    def upload_directory(self, local_dir: str, remote_prefix: str) -> None:
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

    def download_file(self, remote_path: str, local_path: str) -> str:
        source_path = self._resolve(remote_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(source_path, local_path)
        return local_path

    def upload_file(self, local_path: str, remote_path: str) -> None:
        dest_path = self._resolve(remote_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(local_path, dest_path)

    def upload_bytes(self, data: bytes, remote_path: str) -> None:
        dest_path = self._resolve(remote_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(data)

    def upload_directory(self, local_dir: str, remote_prefix: str) -> None:
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


class RemoteDirectoryIterator:
    def __init__(
        self,
        data_loader: DataLoader,
        prefix: str,
        remote_checkpoint_path: str,
        local_checkpoint_path: str,
        local_dir: str,
        batch_size: int = 1000,
    ) -> None:
        self.data_loader = data_loader
        self.prefix = prefix
        self.remote_checkpoint_path = remote_checkpoint_path
        self.local_dir = local_dir
        self.local_checkpoint_path = local_checkpoint_path
        self.batch_size = batch_size
        self.finished: bool = False
        self._continuation_token: str | None = None
        self._load_checkpoint_from_remote()

    def _load_checkpoint_from_remote(self) -> None:
        if self.data_loader.exists(self.remote_checkpoint_path):
            self.data_loader.download_file(
                self.remote_checkpoint_path, self.local_checkpoint_path
            )
            with open(self.local_checkpoint_path) as f:
                checkpoint_data = json.load(f)
                self._continuation_token = checkpoint_data.get("continuation_token")
        else:
            self._continuation_token = None

    def save_checkpoint(self) -> None:
        checkpoint_dir = os.path.dirname(self.local_checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(self.local_checkpoint_path, "w") as f:
            json.dump({"continuation_token": self._continuation_token}, f)
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
                max_keys=min(1000, remaining),
                continuation_token=self._continuation_token,
            )
            self._continuation_token = result.continuation_token
            keys = result.keys
            if filter_fn:
                keys = [key for key in keys if filter_fn(key)]
            if keys:
                pairs = [
                    (key, self._build_local_path(self.prefix, key, self.local_dir))
                    for key in keys
                ]
                downloaded_paths.extend(
                    self._download_files_parallel(pairs, num_workers=num_workers)
                )
                remaining = max_keys - len(downloaded_paths)
                if remaining <= 0:
                    break
            if not result.is_truncated:
                self.finished = True
                break
        return downloaded_paths

    def _download_files_parallel(
        self,
        pairs: list[tuple[str, str]],
        num_workers: int | None = None,
    ) -> list[str]:
        if num_workers is None:
            num_workers = min(10, (os.cpu_count() or 1) * 5)

        max_workers = min(num_workers, len(pairs))
        remote_paths = [pair[0] for pair in pairs]
        local_paths = [pair[1] for pair in pairs]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(self.data_loader.download_file, remote_paths, local_paths)
            )
