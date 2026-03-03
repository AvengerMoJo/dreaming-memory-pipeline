"""
Dreaming Pipeline - A→B→C→D Executor

Orchestrates the full memory consolidation workflow:
- A: Raw conversation input
- B: Semantic chunking
- C: Synthesis and clustering
- D: Archival and versioning

Supports pluggable storage backends via StorageBackend ABC.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from dreaming.models import BChunk, CCluster
from dreaming.chunker import ConversationChunker
from dreaming.synthesizer import DreamingSynthesizer
from dreaming.storage.base import StorageBackend
from dreaming.storage.json_backend import JsonFileBackend

logger = logging.getLogger(__name__)


class DreamingPipeline:
    """
    Full A→B→C→D dreaming pipeline executor

    Transforms raw conversations into consolidated knowledge base
    """

    def __init__(
        self,
        llm_interface,
        quality_level: str = "basic",
        storage: Optional[StorageBackend] = None,
        storage_path: Optional[Path] = None,
        logger=None
    ):
        """
        Initialize pipeline

        Args:
            llm_interface: LLM interface instance (any object with generate_response())
            quality_level: Target quality (basic/good/premium)
            storage: Pluggable storage backend (default: JsonFileBackend)
            storage_path: Path for JsonFileBackend (ignored if storage is provided)
            logger: Optional logger instance
        """
        self.llm = llm_interface
        self.quality_level = quality_level
        self._logger = logger

        # Storage backend
        if storage is not None:
            self.storage = storage
        else:
            self.storage = JsonFileBackend(storage_path=storage_path)

        # Initialize components
        self.chunker = ConversationChunker(
            llm_interface=llm_interface,
            quality_level=quality_level,
            logger=logger
        )
        self.synthesizer = DreamingSynthesizer(
            llm_interface=llm_interface,
            quality_level=quality_level,
            logger=logger
        )

    def _log(self, message: str, level: str = "info"):
        """Log message if logger available"""
        if self._logger:
            getattr(self._logger, level)(f"[DreamingPipeline] {message}")

    def get_manifest(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Public read-only manifest accessor."""
        return self.storage.get_manifest(conversation_id)

    def _update_manifest_for_new_version(
        self,
        conversation_id: str,
        new_version: int,
        previous_version: Optional[int],
    ) -> None:
        """Update lifecycle/lineage metadata in manifest for a newly created version."""
        manifest = self.storage.get_manifest(conversation_id) or {
            "conversation_id": conversation_id,
            "versions": {},
        }
        versions = manifest.setdefault("versions", {})

        # Demote previous latest
        if previous_version is not None:
            prev_key = str(previous_version)
            prev = versions.get(prev_key, {})
            prev["is_latest"] = False
            prev["status"] = "superseded"
            prev["storage_location"] = "cold"
            prev["superseded_by_version"] = new_version
            prev["superseded_at"] = datetime.now().isoformat()
            versions[prev_key] = prev

        new_key = str(new_version)
        versions[new_key] = {
            "is_latest": True,
            "status": "active",
            "storage_location": "hot",
            "previous_version": previous_version,
            "supersedes_version": previous_version,
        }

        manifest["latest_version"] = new_version
        manifest["updated_at"] = datetime.now().isoformat()
        self.storage.update_manifest(conversation_id, manifest)

    def get_archive_lifecycle(
        self, conversation_id: str, version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Get lifecycle/lineage metadata for an archive version from manifest."""
        manifest = self.get_manifest(conversation_id)
        if not manifest:
            return None

        if version is None:
            version = int(manifest.get("latest_version", 0))

        versions = manifest.get("versions", {})
        lifecycle = versions.get(str(version))
        if not lifecycle:
            return None

        return {
            "conversation_id": conversation_id,
            "version": version,
            **lifecycle,
        }

    async def process_conversation(
        self,
        conversation_id: str,
        conversation_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single conversation through A→B→C→D pipeline

        Args:
            conversation_id: Unique conversation identifier
            conversation_text: Raw conversation content (A)
            metadata: Optional metadata for the conversation

        Returns:
            Dict with processing results and paths to stored artifacts
        """
        self._log(f"Processing conversation: {conversation_id}")

        metadata = metadata or {}
        results = {
            "conversation_id": conversation_id,
            "quality_level": self.quality_level,
            "started_at": datetime.now().isoformat(),
            "stages": {}
        }

        try:
            # Stage 1: A→B (Chunking)
            self._log("Stage A→B: Chunking conversation")
            b_chunks = await self.chunker.chunk_conversation(
                conversation_id=conversation_id,
                conversation_text=conversation_text,
                metadata=metadata
            )
            results["stages"]["B_chunks"] = {
                "count": len(b_chunks),
                "chunk_ids": [c.id for c in b_chunks]
            }
            self._log(f"Created {len(b_chunks)} B chunks")

            # Stage 2: B→C (Synthesis)
            self._log("Stage B→C: Synthesizing clusters")
            c_clusters = await self.synthesizer.synthesize_chunks(
                chunks=b_chunks,
                session_id=conversation_id,
                metadata=metadata
            )
            results["stages"]["C_clusters"] = {
                "count": len(c_clusters),
                "cluster_ids": [c.id for c in c_clusters],
                "types": [c.cluster_type.value for c in c_clusters]
            }
            self._log(f"Created {len(c_clusters)} C clusters")

            # Stage 3: C→D (Archival)
            self._log("Stage C→D: Archiving knowledge")
            latest_version = self.storage.get_latest_version(conversation_id)
            previous_version = latest_version if latest_version > 0 else None
            next_version = latest_version + 1

            archive_data = self._create_archive_data(
                conversation_id=conversation_id,
                version=next_version,
                previous_version=previous_version,
                b_chunks=b_chunks,
                c_clusters=c_clusters,
                metadata=metadata
            )

            # Save via storage backend
            self.storage.save_archive(conversation_id, next_version, archive_data)
            self._update_manifest_for_new_version(
                conversation_id=conversation_id,
                new_version=next_version,
                previous_version=previous_version,
            )

            results["stages"]["D_archive"] = {
                "archive_id": archive_data["id"],
                "version": archive_data["version"],
                "entities_count": len(archive_data["entities"]),
                "relationships_count": len(archive_data["relationships"]),
                "previous_version": archive_data["metadata"].get("previous_version"),
                "supersedes_version": archive_data["metadata"].get("supersedes_version"),
                "status": archive_data["metadata"].get("status"),
                "storage_location": archive_data["metadata"].get("storage_location"),
            }
            self._log(f"Archived version {next_version}")

            results["completed_at"] = datetime.now().isoformat()
            results["status"] = "success"

            return results

        except Exception as e:
            self._log(f"Pipeline error: {e}", "error")
            results["completed_at"] = datetime.now().isoformat()
            results["status"] = "failed"
            results["error"] = str(e)
            return results

    def _create_archive_data(
        self,
        conversation_id: str,
        version: int,
        previous_version: Optional[int],
        b_chunks: List[BChunk],
        c_clusters: List[CCluster],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create serialized archive dict from B chunks and C clusters"""

        # Collect all entities
        all_entities = set()
        relationships = []

        for chunk in b_chunks:
            all_entities.update(chunk.entities)

        for cluster in c_clusters:
            for chunk_id in cluster.related_chunks:
                chunk = next((c for c in b_chunks if c.id == chunk_id), None)
                if chunk:
                    all_entities.update(chunk.entities)

        archive_id = f"d_{conversation_id}"
        return {
            "id": archive_id,
            "conversation_id": conversation_id,
            "version": version,
            "quality_level": self.quality_level,
            "created_at": datetime.now().isoformat(),
            "entities": list(all_entities),
            "relationships": relationships[:100],
            "metadata": {
                **metadata,
                "previous_version": previous_version,
                "supersedes_version": previous_version,
                "is_latest": True,
                "status": "active",
                "storage_location": "hot",
            },
            "b_chunks": [
                {
                    "id": c.id,
                    "content": c.content,
                    "labels": c.labels,
                    "speaker": c.speaker,
                    "entities": c.entities,
                    "confidence": c.confidence
                }
                for c in b_chunks
            ],
            "c_clusters": [
                {
                    "id": c.id,
                    "cluster_type": c.cluster_type.value,
                    "theme": c.theme,
                    "content": c.content,
                    "related_chunks": c.related_chunks,
                    "confidence": c.confidence
                }
                for c in c_clusters
            ]
        }

    async def upgrade_quality(
        self,
        conversation_id: str,
        target_quality: str = "good"
    ) -> Dict[str, Any]:
        """
        Upgrade existing archive to higher quality

        Args:
            conversation_id: ID of conversation to upgrade
            target_quality: Target quality level (good/premium)

        Returns:
            Dict with upgrade results
        """
        self._log(f"Upgrading {conversation_id} to {target_quality} quality")

        # Load existing archive
        archive_data = self.storage.load_archive(conversation_id)
        if archive_data is None:
            raise ValueError(f"Archive not found for {conversation_id}")

        # Extract original conversation
        original_text = archive_data.get("metadata", {}).get("original_text", "")
        if not original_text:
            raise ValueError("Original text not found in archive metadata")

        # Re-process with higher quality
        old_quality = self.quality_level
        self.quality_level = target_quality
        self.chunker.quality_level = target_quality
        self.synthesizer.quality_level = target_quality

        try:
            results = await self.process_conversation(
                conversation_id=conversation_id,
                conversation_text=original_text,
                metadata=archive_data.get("metadata", {})
            )

            results["upgraded_from"] = old_quality
            results["upgraded_to"] = target_quality

            return results

        finally:
            self.quality_level = old_quality
            self.chunker.quality_level = old_quality
            self.synthesizer.quality_level = old_quality

    def get_archive(self, conversation_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Retrieve archive from storage"""
        return self.storage.load_archive(conversation_id, version)

    def list_archives(self) -> List[Dict[str, Any]]:
        """List all available archives"""
        return self.storage.list_archives()
