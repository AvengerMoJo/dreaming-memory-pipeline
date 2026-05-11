"""
Backend registry for mojo_memory storage.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

from mojo_memory.storage.base import StorageBackend
from mojo_memory.storage.duckdb_backend import DuckDBStorageBackend
from mojo_memory.storage.local_fs_backend import LocalFileStorageBackend
from mojo_memory.storage.mirror_backend import MirrorStorageBackend

BackendFactory = Callable[..., StorageBackend]

_REGISTRY: Dict[str, BackendFactory] = {
    "local_fs": LocalFileStorageBackend,
    "duckdb": DuckDBStorageBackend,
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
    if name == "mirror":
        primary_cfg = kwargs.get("primary")
        if not isinstance(primary_cfg, dict) or "name" not in primary_cfg:
            raise ValueError("mirror backend requires primary={name, config?}")
        primary_name = primary_cfg["name"]
        primary_kwargs = primary_cfg.get("config", {})
        primary = create_storage_backend(primary_name, **primary_kwargs)

        mirrors_cfg = kwargs.get("mirrors", [])
        mirrors = []
        for cfg in mirrors_cfg:
            mirror_name = cfg["name"]
            mirror_kwargs = cfg.get("config", {})
            mirrors.append(create_storage_backend(mirror_name, **mirror_kwargs))
        return MirrorStorageBackend(
            primary=primary,
            mirrors=mirrors,
            compare_on_read=bool(kwargs.get("compare_on_read", False)),
        )

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
