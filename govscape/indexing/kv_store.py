import os
from collections.abc import Iterable
from typing import Any

# Open LMDB environments by path, with the process that opened each.
_ENVIRONMENTS: dict[str, tuple[Any, int]] = {}


class KVStore:
    """A persistent bytes -> bytes map in an LMDB environment at path.

    The environment is opened lazily, on the first read or write, never on
    construction: the server loads its indexes before gunicorn forks its workers,
    and an LMDB environment opened before fork() can't be used in the child (doing
    so raises an error). LMDB also forbids opening an environment twice in a
    process, so stores on the same path share one.
    """

    def __init__(self, path: str, map_size: int = 1 << 40):
        self.path = path
        # Only reserves address space; the file grows as data is added.
        self.map_size = map_size
        self._env = None
        self._env_pid: int | None = None

    def __getstate__(self):
        return {**self.__dict__, "_env": None, "_env_pid": None}

    def _open(self):
        path = os.path.realpath(self.path)
        if self._env is None:
            import lmdb

            if path not in _ENVIRONMENTS:
                os.makedirs(path, exist_ok=True)
                _ENVIRONMENTS[path] = (
                    lmdb.open(path, map_size=self.map_size),
                    os.getpid(),
                )
            self._env, self._env_pid = _ENVIRONMENTS[path]
        if self._env_pid != os.getpid():
            raise RuntimeError(
                f"LMDB store {path} was opened before fork(); it must be opened "
                "lazily in each worker process."
            )
        return self._env

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def get_many(self, keys: Iterable[bytes]) -> list[bytes | None]:
        with self._open().begin() as txn:
            return [txn.get(key) for key in keys]

    def put_many(self, items: Iterable[tuple[bytes, bytes]], append=False) -> None:
        """Write the (key, value) pairs in one transaction. append=True is much
        faster but needs keys in sorted order, after any already stored."""
        with self._open().begin(write=True) as txn:
            txn.cursor().putmulti(items, append=append)

    def sync(self) -> None:
        if self._env is not None and self._env_pid == os.getpid():
            self._env.sync()

    def close(self) -> None:
        if self._env is not None and self._env_pid == os.getpid():
            self._env.close()
            _ENVIRONMENTS.pop(os.path.realpath(self.path), None)
        self._env = None
        self._env_pid = None
