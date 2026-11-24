import importlib
import sys
from collections.abc import Callable
from typing import Any


def lazy_getattr(package: str, exports: dict[str, str]) -> Callable[[str], Any]:
    """Return a module __getattr__ for package that imports each name in exports
    from its submodule (exports maps name -> submodule) on first use."""

    def __getattr__(name: str) -> Any:
        if name not in exports:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(f".{exports[name]}", package), name)
        setattr(sys.modules[package], name, value)
        return value

    return __getattr__
