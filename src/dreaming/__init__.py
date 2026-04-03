"""
Dreaming Memory Pipeline

Two ingestion paths:

  process_conversation()  — A→B→C→D for task sessions and chat histories.
                            Segments text into semantic chunks (B), clusters
                            them (C), and archives (D).

  process_document()      — atomic fact extraction for research reports,
                            articles, and design docs. Extracts KnowledgeUnits
                            (one proposition + source quote each), computes
                            inter-unit links, and archives directly (D).

Supports pluggable storage backends (JSON files, SQLite, PostgreSQL, etc.)
"""

from dreaming.models import BChunk, CCluster, DArchive, KnowledgeUnit
from dreaming.pipeline import DreamingPipeline
from dreaming.chunker import ConversationChunker
from dreaming.synthesizer import DreamingSynthesizer
from dreaming.atomic_extractor import AtomicFactExtractor
from dreaming.storage.base import StorageBackend
from dreaming.storage.json_backend import JsonFileBackend

__all__ = [
    'BChunk',
    'CCluster',
    'DArchive',
    'KnowledgeUnit',
    'DreamingPipeline',
    'ConversationChunker',
    'DreamingSynthesizer',
    'AtomicFactExtractor',
    'StorageBackend',
    'JsonFileBackend',
]
