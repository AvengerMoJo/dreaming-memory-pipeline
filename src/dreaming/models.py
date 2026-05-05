"""
Dreaming Data Models

Data structures for A→B→C→D memory consolidation pipeline.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class ChunkType(Enum):
    """Type of B chunk"""
    SEMANTIC = "semantic"  # Topic/idea boundaries
    SPEAKER_TURN = "speaker_turn"  # Each speaker's contribution
    ENTITY = "entity"  # Named entity occurrence
    RELATIONSHIP = "relationship"  # Connection between entities


class ClusterType(Enum):
    """Type of C cluster"""
    TOPIC = "topic"  # Grouped by topic/theme
    RELATIONSHIP = "relationship"  # Explicit connections
    SUMMARY = "summary"  # High-level overview
    TIMELINE = "timeline"  # Chronological progression
    DECISION = "decision"  # Explicit decisions made (e.g., chose Docker over bash_exec)
    OUTCOME = "outcome"  # Concrete results (e.g., briefing saved to ~/.memory/scout_briefing.md)
    FINDING = "finding"  # Discovered facts (e.g., Rowhammer GPU attacks give complete control)
    TOOL = "tool"  # Tool usage patterns (e.g., cella should be exercised through tmux)
    INCOMPLETE = "incomplete"  # Unresolved items that need follow-up


class ArchiveStatus(Enum):
    """Status of D archive"""
    SUPERSEDED = "superseded"  # Replaced by newer version
    DUPLICATE = "duplicate"  # Exact or near-duplicate content
    OBSOLETE = "obsolete"  # Outdated information
    HISTORICAL = "historical"  # Kept for reference only


@dataclass
class BChunk:
    """
    Deconstructed semantic chunk (B)

    Created from raw conversation data (A) with rich metadata
    """

    # Required fields
    id: str
    parent_id: str  # Link to source A chunk
    chunk_type: ChunkType
    content: str

    # Metadata
    labels: List[str] = field(default_factory=list)
    speaker: Optional[str] = None  # user/assistant/system
    entities: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)  # Atomic facts extracted from chunk
    confidence: float = 1.0  # AI confidence (0-1)

    # Position tracking
    token_range: tuple[int, int] = (0, 0)  # (start, end) in parent
    position_in_parent: float = 0.0  # Relative position (0-1)

    # Embedding
    embedding: Optional[List[float]] = None

    # Quality tracking
    quality_level: str = "basic"  # basic/good/premium
    needs_upgrade: bool = True
    llm_used: Optional[str] = None
    language: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        data = asdict(self)
        data['chunk_type'] = self.chunk_type.value
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BChunk':
        """Create from dictionary"""
        data['chunk_type'] = ChunkType(data['chunk_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class CCluster:
    """
    Synthesized cluster (C)

    Global consolidated view combining multiple B chunks
    """

    # Required fields
    id: str
    cluster_type: ClusterType
    content: str  # Summary or consolidated content

    # Relationships
    related_chunks: List[str] = field(default_factory=list)  # B chunk IDs
    related_clusters: List[str] = field(default_factory=list)  # Other C IDs

    # Metadata
    theme: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    confidence: float = 1.0
    confidence_level: str = "medium"  # high/medium/low confidence annotation
    key_facts: List[str] = field(default_factory=list)  # Atomic facts from cluster

    # Temporal span
    time_span_start: Optional[datetime] = None
    time_span_end: Optional[datetime] = None

    # Contradiction resolution
    contradictions_resolved: List[str] = field(default_factory=list)

    # Embedding
    embedding: Optional[List[float]] = None

    # Version tracking
    version: int = 1

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        data = asdict(self)
        data['cluster_type'] = self.cluster_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.time_span_start:
            data['time_span_start'] = self.time_span_start.isoformat()
        if self.time_span_end:
            data['time_span_end'] = self.time_span_end.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CCluster':
        """Create from dictionary"""
        data['cluster_type'] = ClusterType(data['cluster_type'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('time_span_start'):
            data['time_span_start'] = datetime.fromisoformat(data['time_span_start'])
        if data.get('time_span_end'):
            data['time_span_end'] = datetime.fromisoformat(data['time_span_end'])
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class KnowledgeUnit:
    """
    Atomic knowledge unit (D)

    A single self-contained proposition, independently meaningful
    """

    # Required fields
    id: str
    content: str  # The atomic fact / proposition
    source_doc_id: str  # Which document it came from

    # Anchoring
    quote: Optional[str] = None  # Direct quote from source text

    # Metadata
    labels: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    confidence: float = 0.5

    # Links to related units
    links: List[str] = field(default_factory=list)  # IDs of related KnowledgeUnits

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeUnit':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DArchive:
    """
    Consolidated knowledge archive (D)

    Final output: B chunks + C clusters + KnowledgeUnits, archived with versioning
    """

    # Required fields
    id: str
    conversation_id: str  # Source A chunk ID
    version: int  # Archive version number

    # Content
    b_chunks: List[BChunk] = field(default_factory=list)
    c_clusters: List[CCluster] = field(default_factory=list)
    knowledge_units: List[KnowledgeUnit] = field(default_factory=list)

    # Metadata
    quality_level: str = "basic"
    entities: List[str] = field(default_factory=list)  # All entities across chunks/clusters
    relationships: List[str] = field(default_factory=list)  # Cross-references between clusters

    # Versioning
    previous_version: Optional[int] = None
    superseded_by_version: Optional[int] = None
    is_latest: bool = True
    status: str = "active"
    storage_location: str = "hot"

    # Archive metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['b_chunks'] = [c.to_dict() for c in self.b_chunks]
        data['c_clusters'] = [c.to_dict() for c in self.c_clusters]
        data['knowledge_units'] = [u.to_dict() for u in self.knowledge_units]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DArchive':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['b_chunks'] = [BChunk.from_dict(c) for c in data.get('b_chunks', [])]
        data['c_clusters'] = [CCluster.from_dict(c) for c in data.get('c_clusters', [])]
        data['knowledge_units'] = [KnowledgeUnit.from_dict(u) for u in data.get('knowledge_units', [])]
        return cls(**data)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class DreamingStats:
    """Summary statistics for a dreaming pipeline run"""

    def __init__(
        self,
        chunks: int = 0,
        clusters: int = 0,
        entities: int = 0,
        knowledge_units: int = 0,
        quality_level: str = "basic",
        quality_flags: Optional[List[str]] = None,
        status: str = "success"
    ):
        self.chunks = chunks
        self.clusters = clusters
        self.entities = entities
        self.knowledge_units = knowledge_units
        self.quality_level = quality_level
        self.quality_flags = quality_flags or []
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": self.chunks,
            "clusters": self.clusters,
            "entities": self.entities,
            "knowledge_units": self.knowledge_units,
            "quality_level": self.quality_level,
            "quality_flags": self.quality_flags,
            "status": self.status,
        }
