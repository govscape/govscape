from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Callable, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from botocore.client import BaseClient as S3Client

@dataclass
class ListResult:
    keys: List[str]
    is_truncated: bool
    next_token: Optional[str]


class DataLoader(ABC):
    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path
        self._continuation_token: Optional[str] = None
        self._checkpoint_loaded = False
        self._checkpoint_prefix: Optional[str] = None
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
            return
        try:
            with open(self.checkpoint_path, "r") as f:
                payload = json.load(f)
            token = payload.get("continuation_token")
            if token:
                self._continuation_token = token
            self._checkpoint_loaded = True
        except Exception:
            return

    def _checkpoint_filename(self) -> Optional[str]:
        if not self.checkpoint_path:
            return None
        return os.path.basename(self.checkpoint_path)

    def _checkpoint_remote_path(self, prefix: str) -> Optional[str]:
        filename = self._checkpoint_filename()
        if not filename:
            return None
        normalized_prefix = prefix.rstrip("/")
        if not normalized_prefix:
            return filename
        return f"{normalized_prefix}/{filename}"

    def _ensure_checkpoint_loaded(self, prefix: str) -> None:
        if not self.checkpoint_path or self._checkpoint_loaded:
            if prefix:
                self._checkpoint_prefix = prefix
            return
        if prefix:
            self._checkpoint_prefix = prefix
        self._load_checkpoint()
        if self._checkpoint_loaded:
            return
        self._load_checkpoint_from_remote(prefix)
        self._checkpoint_loaded = True

    def _load_checkpoint_from_remote(self, prefix: str) -> None:
        return

    def _save_checkpoint_to_remote(self, prefix: str) -> None:
        return

    @staticmethod
    def _build_local_path(prefix: str, remote_path: str, local_dir: str) -> str:
        rel_path = remote_path
        if prefix and remote_path.startswith(prefix):
            rel_path = remote_path[len(prefix) :]
        rel_path = rel_path.lstrip("/")
        if not rel_path:
            rel_path = os.path.basename(remote_path)
        return os.path.join(local_dir, rel_path)

    def download_files(
        self,
        prefix: str,
        local_dir: str,
        max_keys: int = 1000,
        filter_fn: Optional[Callable[[str], bool]] = None,
        num_workers: Optional[int] = None,
    ) -> List[str]:
        downloaded_paths: List[str] = []
        remaining = max_keys
        while remaining > 0:
            result = self.list_objects(prefix, max_keys=min(1000, remaining))
            keys = result.keys
            if filter_fn:
                keys = [key for key in keys if filter_fn(key)]
            if keys:
                pairs = [(key, self._build_local_path(prefix, key, local_dir)) for key in keys]
                downloaded_paths.extend(
                    self._download_files_parallel(pairs, num_workers=num_workers)
                )
                remaining = max_keys - len(downloaded_paths)
                if remaining <= 0:
                    break
            if not result.is_truncated:
                break
        return downloaded_paths

    def _download_files_parallel(
        self,
        pairs: List[tuple[str, str]],
        num_workers: Optional[int] = None,
    ) -> List[str]:
        if not pairs:
            return []
        if num_workers is None:
            num_workers = min(10, (os.cpu_count() or 1) * 5)

        max_workers = min(num_workers, len(pairs))
        downloaded: List[str] = []
        remote_paths = [pair[0] for pair in pairs]
        local_paths = [pair[1] for pair in pairs]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for local_path in executor.map(self._download_single, remote_paths, local_paths):
                downloaded.append(local_path)
        return downloaded

    def _download_single(self, remote_path: str, local_path: str) -> str:
        self.download_file(remote_path, local_path)
        return local_path

    @abstractmethod
    def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000,
    ) -> ListResult:
        raise NotImplementedError

    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> None:
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

    @abstractmethod
    def save_checkpoint(self) -> None:
        raise NotImplementedError


class S3DataLoader(DataLoader):
    def __init__(
        self,
        bucket_name: str,
        config: Optional[Config] = Config(max_pool_connections=60),
        s3_client: Optional[S3Client] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path)
        self.bucket_name = bucket_name
        self.s3 : S3Client = s3_client or boto3.client("s3", config=config)

    def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000,
    ) -> ListResult:
        self._ensure_checkpoint_loaded(prefix)
        kwargs = {
            "Bucket": self.bucket_name,
            "Prefix": prefix,
            "MaxKeys": max_keys,
        }
        if self._continuation_token:
            kwargs["ContinuationToken"] = self._continuation_token
        result = self.s3.list_objects_v2(**kwargs)
        contents = result.get("Contents", [])
        keys = [obj["Key"] for obj in contents]
        checkpoint_remote = self._checkpoint_remote_path(prefix)
        if checkpoint_remote:
            keys = [key for key in keys if key != checkpoint_remote]
        self._continuation_token = result.get("NextContinuationToken")
        return ListResult(
            keys=keys,
            is_truncated=result.get("IsTruncated", False),
            next_token=self._continuation_token,
        )

    def download_file(self, remote_path: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket_name, remote_path, local_path)

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

    def save_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump({"continuation_token": self._continuation_token}, f)
        if self._checkpoint_prefix:
            self._save_checkpoint_to_remote(self._checkpoint_prefix)

    def _load_checkpoint_from_remote(self, prefix: str) -> None:
        remote_path = self._checkpoint_remote_path(prefix)
        if not remote_path or not self.checkpoint_path:
            return
        try:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            self.s3.download_file(self.bucket_name, remote_path, self.checkpoint_path)
            self._load_checkpoint()
        except Exception:
            return

    def _save_checkpoint_to_remote(self, prefix: str) -> None:
        remote_path = self._checkpoint_remote_path(prefix)
        if not remote_path or not self.checkpoint_path:
            return
        try:
            self.s3.upload_file(self.checkpoint_path, self.bucket_name, remote_path)
        except Exception:
            return


class LocalDataLoader(DataLoader):
    def __init__(self, base_dir: str, checkpoint_path: Optional[str] = None) -> None:
        super().__init__(checkpoint_path=checkpoint_path)
        self.base_dir = os.path.abspath(base_dir)

    def _resolve(self, remote_path: str) -> str:
        return os.path.join(self.base_dir, remote_path)

    def list_objects(
        self,
        prefix: str,
        max_keys: int = 1000,
    ) -> ListResult:
        self._ensure_checkpoint_loaded(prefix)
        start_index = int(self._continuation_token) if self._continuation_token else 0
        base_path = self._resolve(prefix)
        keys: List[str] = []
        if os.path.exists(base_path):
            for root, _, files in os.walk(base_path):
                for filename in files:
                    abs_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(abs_path, self.base_dir)
                    keys.append(rel_path.replace("\\", "/"))
        checkpoint_remote = self._checkpoint_remote_path(prefix)
        if checkpoint_remote:
            keys = [key for key in keys if key != checkpoint_remote]
        keys.sort()
        slice_keys = keys[start_index : start_index + max_keys]
        next_index = start_index + max_keys
        next_token = str(next_index) if next_index < len(keys) else None
        self._continuation_token = next_token
        return ListResult(
            keys=slice_keys,
            is_truncated=next_token is not None,
            next_token=next_token,
        )

    def download_file(self, remote_path: str, local_path: str) -> None:
        source_path = self._resolve(remote_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(source_path, local_path)

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
        try:
            os.remove(self._resolve(remote_path))
        except FileNotFoundError:
            pass

    def exists(self, remote_path: str) -> bool:
        return os.path.exists(self._resolve(remote_path))

    def to_uri(self, remote_path: str) -> str:
        return f"file://{self._resolve(remote_path)}"

    def save_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        checkpoint_dir = os.path.dirname(self.checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        with open(self.checkpoint_path, "w") as f:
            json.dump({"continuation_token": self._continuation_token}, f)
        if self._checkpoint_prefix:
            self._save_checkpoint_to_remote(self._checkpoint_prefix)

    def _load_checkpoint_from_remote(self, prefix: str) -> None:
        remote_path = self._checkpoint_remote_path(prefix)
        if not remote_path or not self.checkpoint_path:
            return
        remote_abs = self._resolve(remote_path)
        if not os.path.exists(remote_abs):
            return
        try:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            shutil.copy2(remote_abs, self.checkpoint_path)
            self._load_checkpoint()
        except Exception:
            return

    def _save_checkpoint_to_remote(self, prefix: str) -> None:
        remote_path = self._checkpoint_remote_path(prefix)
        if not remote_path or not self.checkpoint_path:
            return
        try:
            remote_abs = self._resolve(remote_path)
            os.makedirs(os.path.dirname(remote_abs), exist_ok=True)
            shutil.copy2(self.checkpoint_path, remote_abs)
        except Exception:
            return


def build_data_loader(
    backend: str,
    bucket_name: Optional[str] = None,
    local_base_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    config: Optional[Config] = None,
    s3_client: Optional[S3Client] = None,
) -> DataLoader:
    if backend == "local":
        if not local_base_dir:
            raise ValueError("local_base_dir is required for local backend")
        return LocalDataLoader(local_base_dir, checkpoint_path=checkpoint_path)
    elif backend == "s3":
        if not bucket_name:
            raise ValueError("bucket_name is required for s3 backend")
        return S3DataLoader(
            bucket_name=bucket_name,
            config=config,
            s3_client=s3_client,
            checkpoint_path=checkpoint_path,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
