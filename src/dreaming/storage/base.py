"""
Storage Backend ABC

Abstract base class for archive storage backends.
Implementations can use JSON files, SQLite, PostgreSQL, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class StorageBackend(ABC):
    """Abstract storage backend for dreaming archives"""

    @abstractmethod
    def save_archive(self, conversation_id: str, version: int, data: dict) -> None:
        """
        Save an archive version.

        Args:
            conversation_id: Unique conversation identifier
            version: Archive version number (1-based)
            data: Serialized archive data dict
        """
        ...

    @abstractmethod
    def load_archive(self, conversation_id: str, version: int | None = None) -> dict | None:
        """
        Load an archive. If version is None, load the latest.

        Args:
            conversation_id: Unique conversation identifier
            version: Specific version to load (None = latest)

        Returns:
            Archive data dict, or None if not found
        """
        ...

    @abstractmethod
    def list_archives(self) -> list[dict]:
        """
        List all archived conversations with summary metadata.

        Returns:
            List of dicts with conversation_id, latest_version, quality_level, etc.
        """
        ...

    @abstractmethod
    def get_manifest(self, conversation_id: str) -> dict | None:
        """
        Get version manifest for a conversation.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            Manifest dict with versions, latest_version, etc. or None
        """
        ...

    @abstractmethod
    def update_manifest(self, conversation_id: str, manifest: dict) -> None:
        """
        Save/update manifest for a conversation.

        Args:
            conversation_id: Unique conversation identifier
            manifest: Full manifest dict to save
        """
        ...

    @abstractmethod
    def get_latest_version(self, conversation_id: str) -> int:
        """
        Get the latest archive version number for a conversation.

        Args:
            conversation_id: Unique conversation identifier

        Returns:
            Latest version number (0 if no archives exist)
        """
        ...
