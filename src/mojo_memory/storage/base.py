"""
Generic storage backend contract for mojo_memory persistence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class StorageBackend(ABC):
    """Abstract key-based storage backend for JSON-serializable data."""

    @abstractmethod
    def read_json(self, key: str) -> Any | None:
        """Read JSON value by key. Returns None when key is missing."""
        ...

    @abstractmethod
    def write_json(self, key: str, data: Any) -> None:
        """Write JSON value by key, replacing existing data."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether key exists."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key. Returns True when removed, False when missing."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys, optionally filtered by prefix."""
        ...

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return backend health and diagnostics."""
        ...
