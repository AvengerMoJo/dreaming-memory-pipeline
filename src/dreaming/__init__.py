"""
Dreaming Memory Pipeline

Transforms raw conversations (A) into a consolidated knowledge base through:
- B: Semantic chunking with rich metadata
- C: Global synthesis and clustering
- D: Archival and versioning

Supports pluggable storage backends (JSON files, SQLite, PostgreSQL, etc.)
"""

from dreaming.models import BChunk, CCluster, DArchive
from dreaming.pipeline import DreamingPipeline
from dreaming.chunker import ConversationChunker
from dreaming.synthesizer import DreamingSynthesizer
from dreaming.storage.base import StorageBackend
from dreaming.storage.json_backend import JsonFileBackend

__all__ = [
    'BChunk',
    'CCluster',
    'DArchive',
    'DreamingPipeline',
    'ConversationChunker',
    'DreamingSynthesizer',
    'StorageBackend',
    'JsonFileBackend',
]
