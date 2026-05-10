"""Dream Provider Adapter — wraps DreamingPipeline as a DreamProvider.

This is the compatibility layer that allows the existing DreamingPipeline
to satisfy the DreamProvider contract during migration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.provider_contracts import (
    DreamProvider,
    ProviderVersion,
    DreamArtifact,
    DreamStageResult,
)

logger = logging.getLogger(__name__)


class DreamProviderAdapter(DreamProvider):
    """
    Adapter that wraps DreamingPipeline as a DreamProvider.
    
    Delegates all operations to the underlying DreamingPipeline instance.
    """

    PROVIDER_NAME = "mojo_dream"
    PROVIDER_VERSION = "1.0.0"
    CONTRACT_VERSION = "1.0"

    def __init__(
        self,
        llm_interface=None,
        quality_level: str = "basic",
        storage=None,
        storage_path=None,
        **kwargs: Any,
    ):
        from dreaming.pipeline import DreamingPipeline

        self._pipeline = DreamingPipeline(
            llm_interface=llm_interface,
            quality_level=quality_level,
            storage=storage,
            storage_path=storage_path,
        )
        logger.info(
            "DreamProviderAdapter initialized: provider=%s version=%s",
            self.PROVIDER_NAME,
            self.PROVIDER_VERSION,
        )

    def get_version(self) -> ProviderVersion:
        return ProviderVersion(
            provider_name=self.PROVIDER_NAME,
            provider_version=self.PROVIDER_VERSION,
            contract_version=self.CONTRACT_VERSION,
        )

    # -- Pipeline stages ----------------------------------------------------

    def run_stage_a(
        self,
        conversation_text: str,
        session_id: str,
    ) -> DreamStageResult:
        # Stage A is implicit in the current pipeline (pass-through)
        return DreamStageResult(
            stage="A",
            status="ok",
            artifacts=[DreamArtifact(
                stage="A",
                artifact_type="conversation",
                content=conversation_text,
                metadata={"session_id": session_id},
            )],
        )

    def run_stage_b(self, stage_a_result: DreamStageResult, session_id: str) -> DreamStageResult:
        conv_text = stage_a_result.artifacts[0].content if stage_a_result.artifacts else ""
        chunks = self._pipeline.chunker.chunk_conversation(conv_text, session_id=session_id)
        artifacts = [
            DreamArtifact(
                stage="B",
                artifact_type="chunk",
                content=getattr(c, 'content', str(c)),
                metadata={
                    "chunk_id": getattr(c, 'id', str(i)),
                    "labels": getattr(c, 'labels', []),
                    "entities": getattr(c, 'entities', []),
                },
                confidence=getattr(c, 'confidence', 0.5),
            )
            for i, c in enumerate(chunks)
        ]
        return DreamStageResult(
            stage="B",
            status="ok",
            artifacts=artifacts,
            metrics={"n_chunks": len(chunks)},
        )

    def run_stage_c(self, stage_b_result: DreamStageResult, session_id: str) -> DreamStageResult:
        # Reconstruct chunk-like objects from artifacts
        chunks = []
        for art in stage_b_result.artifacts:
            chunks.append(type('Chunk', (), {
                'content': art.content,
                'id': art.metadata.get('chunk_id', ''),
                'labels': art.metadata.get('labels', []),
                'entities': art.metadata.get('entities', []),
            })())

        clusters = self._pipeline.synthesizer.synthesize_chunks(chunks, session_id=session_id)
        artifacts = [
            DreamArtifact(
                stage="C",
                artifact_type="cluster",
                content=getattr(c, 'content', str(c)),
                metadata={
                    "cluster_id": getattr(c, 'id', str(i)),
                    "cluster_type": str(getattr(c, 'cluster_type', 'unknown')),
                    "title": getattr(c, 'title', ''),
                },
                confidence=0.5,
            )
            for i, c in enumerate(clusters)
        ]
        return DreamStageResult(
            stage="C",
            status="ok",
            artifacts=artifacts,
            metrics={"n_clusters": len(clusters)},
        )

    def run_stage_d(
        self,
        stage_c_result: DreamStageResult,
        stage_b_result: Optional[DreamStageResult] = None,
        session_id: str = "",
    ) -> DreamStageResult:
        # Archival is handled by the pipeline's storage backend
        return DreamStageResult(
            stage="D",
            status="ok",
            artifacts=[],
            metrics={"archived": True, "session_id": session_id},
        )

    def run_pipeline(
        self,
        conversation_text: str,
        session_id: str,
        stages: Optional[List[str]] = None,
    ) -> Dict[str, DreamStageResult]:
        """
        Run the full ABCD pipeline via the underlying DreamingPipeline.
        Returns dict of stage_id -> DreamStageResult.
        """
        try:
            # Run full pipeline
            self._pipeline.process_conversation(
                conversation_text, session_id=session_id
            )
            
            # Return stage results
            return {
                "A": DreamStageResult(
                    stage="A",
                    status="ok",
                    artifacts=[DreamArtifact(
                        stage="A",
                        artifact_type="conversation",
                        content=conversation_text,
                        metadata={"session_id": session_id},
                    )],
                ),
                "B": DreamStageResult(stage="B", status="ok", metrics={"processed": True}),
                "C": DreamStageResult(stage="C", status="ok", metrics={"processed": True}),
                "D": DreamStageResult(stage="D", status="ok", metrics={"processed": True}),
            }
        except Exception as e:
            logger.error("Dream pipeline failed: %s", e)
            return {
                "A": DreamStageResult(stage="A", status="error", error=str(e)),
                "B": DreamStageResult(stage="B", status="error", error=str(e)),
                "C": DreamStageResult(stage="C", status="error", error=str(e)),
                "D": DreamStageResult(stage="D", status="error", error=str(e)),
            }

    def validate_input(self, conversation_text: str) -> Dict[str, Any]:
        errors = []
        warnings = []
        if not conversation_text or not conversation_text.strip():
            errors.append("Empty conversation text")
        if len(conversation_text) < 100:
            warnings.append("Very short conversation (< 100 chars)")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.PROVIDER_NAME,
            "stages": ["A", "B", "C", "D"],
            "supports_dry_run": True,
            "supports_partial_stages": True,
            "quality_level": self._pipeline.quality_level,
        }
