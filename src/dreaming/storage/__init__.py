"""
Storage Backends

Pluggable storage backends for the dreaming pipeline.
"""

from dreaming.storage.base import StorageBackend
from dreaming.storage.json_backend import JsonFileBackend

__all__ = [
    'StorageBackend',
    'JsonFileBackend',
]
