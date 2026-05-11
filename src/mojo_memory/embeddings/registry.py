"""Registry/factory for embedding backends."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict

from mojo_memory.embeddings.backends.base import EmbeddingBackend
from mojo_memory.embeddings.backends.huggingface_backend import HuggingFaceBackend
from mojo_memory.embeddings.backends.local_server_backend import LocalServerBackend
from mojo_memory.embeddings.backends.random_backend import RandomBackend

BackendFactory = Callable[..., EmbeddingBackend]

_REGISTRY: Dict[str, BackendFactory] = {
    "huggingface": HuggingFaceBackend,
    "local": LocalServerBackend,
    "random": RandomBackend,
    "api": LocalServerBackend,  # backward compatibility with existing "api" mode
}


def register_backend(name: str, factory: BackendFactory) -> None:
    _REGISTRY[name] = factory


def get_backend(name: str) -> BackendFactory | None:
    return _REGISTRY.get(name)


def list_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


def create_backend(name: str, **kwargs: Any) -> EmbeddingBackend:
    if name in _REGISTRY:
        return _REGISTRY[name](**kwargs)
    if ":" in name:
        module_name, class_name = name.split(":", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        return cls(**kwargs)
    raise ValueError(f"Unknown embedding backend '{name}'. Known: {', '.join(list_backends())}")
