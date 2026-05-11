"""Storage backend abstractions and registry for mojo_memory."""

from mojo_memory.storage.base import StorageBackend
from mojo_memory.storage.conversation_schema import ConversationRecord, validate_conversation_record
from mojo_memory.storage.duckdb_backend import DuckDBStorageBackend
from mojo_memory.storage.local_fs_backend import LocalFileStorageBackend
from mojo_memory.storage.mirror_backend import MirrorStorageBackend
from mojo_memory.storage.registry import (
    create_storage_backend,
    list_storage_backends,
    register_storage_backend,
)

__all__ = [
    "StorageBackend",
    "ConversationRecord",
    "validate_conversation_record",
    "LocalFileStorageBackend",
    "DuckDBStorageBackend",
    "MirrorStorageBackend",
    "register_storage_backend",
    "list_storage_backends",
    "create_storage_backend",
]
