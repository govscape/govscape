from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass

from botocore.config import Config
from govscape.data_loader import RemoteDirectoryIterator, build_data_loader


@dataclass
class BenchmarkResult:
    backend: str
    compress_upload: bool
    file_size_bytes: int
    num_files: int
    chunk_size: int
    upload_seconds: float
    download_seconds: float
    listed_object_count: int
    downloaded_file_count: int
    throughput_upload_mib_s: float
    throughput_download_mib_s: float


def _create_test_files(base_dir: str, num_files: int, file_size_bytes: int) -> None:
    payload = os.urandom(file_size_bytes)
    for i in range(num_files):
        subdir = os.path.join(base_dir, f"part_{i % 8}")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, f"file_{i:06d}.bin"), "wb") as f:
            f.write(payload)


def _run_one_benchmark(
    *,
    backend: str,
    file_size_bytes: int,
    num_files: int,
    compress_upload: bool,
    chunk_size: int,
    num_workers: int | None,
    use_multiprocessing: bool,
    bucket_name: str | None,
    local_base_dir: str | None,
) -> BenchmarkResult:
    with ExitStack() as stack:
        src_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="bench_src_"))
        dst_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix="bench_dst_"))
        ckpt_dir = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="bench_ckpt_")
        )

        if backend == "local" and not local_base_dir:
            local_base_dir = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="bench_store_")
            )

        loader = build_data_loader(
            backend=backend,
            bucket_name=bucket_name,
            local_base_dir=local_base_dir,
            config=Config(max_pool_connections=60),
        )

        _create_test_files(src_dir, num_files, file_size_bytes)

        run_id = uuid.uuid4().hex[:12]
        remote_prefix = f"benchmarks/data_loader/{run_id}/payload"
        remote_checkpoint_path = f"benchmarks/data_loader/{run_id}/checkpoint.json"
        local_checkpoint_path = os.path.join(ckpt_dir, "checkpoint.json")

        upload_start = time.perf_counter()
        loader.upload_directory(
            src_dir,
            remote_prefix,
            compress=compress_upload,
            chunk_size=chunk_size,
        )
        upload_seconds = time.perf_counter() - upload_start

        listed = loader.list_objects(remote_prefix, max_keys=1_000_000)
        listed_object_count = len(listed.keys)

        with RemoteDirectoryIterator(
            data_loader=loader,
            prefix=remote_prefix,
            remote_checkpoint_path=remote_checkpoint_path,
            local_checkpoint_path=local_checkpoint_path,
            local_dir=dst_dir,
            use_multiprocessing=use_multiprocessing,
        ) as iterator:
            download_start = time.perf_counter()
            downloaded_paths = iterator.download_batch(
                max_keys=max(1_000_000, num_files),
                num_workers=num_workers,
            )
            iterator.save_checkpoint()
            download_seconds = time.perf_counter() - download_start

        total_mib = (file_size_bytes * num_files) / (1024 * 1024)
        upload_tp = total_mib / upload_seconds if upload_seconds > 0 else 0.0
        download_tp = total_mib / download_seconds if download_seconds > 0 else 0.0

        return BenchmarkResult(
            backend=backend,
            compress_upload=compress_upload,
            file_size_bytes=file_size_bytes,
            num_files=num_files,
            chunk_size=chunk_size,
            upload_seconds=upload_seconds,
            download_seconds=download_seconds,
            listed_object_count=listed_object_count,
            downloaded_file_count=len(downloaded_paths),
            throughput_upload_mib_s=upload_tp,
            throughput_download_mib_s=download_tp,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DataLoader and iterator.")
    parser.add_argument("--backend", choices=["local", "s3"], default="local")
    parser.add_argument("--bucket-name", default=None)
    parser.add_argument("--local-base-dir", default=None)
    parser.add_argument("--file-size-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--num-files", type=int, default=1000)
    parser.add_argument("--compress-upload", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--disable-multiprocessing", action="store_true")
    args = parser.parse_args()

    result = _run_one_benchmark(
        backend=args.backend,
        file_size_bytes=args.file_size_bytes,
        num_files=args.num_files,
        compress_upload=args.compress_upload,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        use_multiprocessing=not args.disable_multiprocessing,
        bucket_name=args.bucket_name,
        local_base_dir=args.local_base_dir,
    )

    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
