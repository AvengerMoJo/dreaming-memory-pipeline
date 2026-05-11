"""
Backend registry for mojo_memory storage.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

from mojo_memory.storage.base import StorageBackend
from mojo_memory.storage.local_fs_backend import LocalFileStorageBackend

BackendFactory = Callable[..., StorageBackend]

_REGISTRY: Dict[str, BackendFactory] = {
    "local_fs": LocalFileStorageBackend,
}


def register_storage_backend(name: str, factory: BackendFactory) -> None:
    """Register a storage backend factory under a stable name."""
    _REGISTRY[name] = factory


def list_storage_backends() -> list[str]:
    """List registered storage backend names."""
    return sorted(_REGISTRY.keys())


def create_storage_backend(name: str, **kwargs: Any) -> StorageBackend:
    """
    Create backend by registered name or import path.

    Supported forms:
    - "local_fs" (registered name)
    - "pkg.module:ClassName" (dynamic import)
    """
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)

    if ":" in name:
        module_name, class_name = name.split(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls(**kwargs)

    raise ValueError(
        f"Unknown storage backend '{name}'. "
        f"Known backends: {', '.join(list_storage_backends())}"
    )
