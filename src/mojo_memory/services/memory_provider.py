"""Memory Provider Adapter — wraps MemoryService as a MemoryProvider.

This is the compatibility layer that allows the existing MemoryService
to satisfy the MemoryProvider contract during migration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.services.provider_contracts import MemoryProvider, ProviderVersion

logger = logging.getLogger(__name__)


class MemoryProviderAdapter(MemoryProvider):
    """
    Adapter that wraps MemoryService as a MemoryProvider.
    
    Delegates all operations to the underlying MemoryService instance.
    """

    PROVIDER_NAME = "mojo_memory"
    PROVIDER_VERSION = "1.0.0"
    CONTRACT_VERSION = "1.0"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        embedding_model: str = "nomic-ai/nomic-embed-text-v2-moe",
        embedding_backend: str = "huggingface",
        embedding_device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        from mojo_memory.services.memory_service import MemoryService

        self._service = MemoryService(
            data_dir=data_dir,
            embedding_model=embedding_model,
            embedding_backend=embedding_backend,
            embedding_device=embedding_device,
            config=config,
        )
        logger.info(
            "MemoryProviderAdapter initialized: provider=%s version=%s",
            self.PROVIDER_NAME,
            self.PROVIDER_VERSION,
        )

    def get_version(self) -> ProviderVersion:
        return ProviderVersion(
            provider_name=self.PROVIDER_NAME,
            provider_version=self.PROVIDER_VERSION,
            contract_version=self.CONTRACT_VERSION,
        )

    # -- Conversation CRUD --------------------------------------------------

    def add_conversation(
        self,
        role_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._service.add_conversation(role_id, content, metadata)

    def get_conversation(
        self,
        role_id: str,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        return self._service.get_conversation(role_id, conversation_id)

    def search_conversations(
        self,
        role_id: str,
        query: str,
        max_items: int = 10,
    ) -> List[Dict[str, Any]]:
        return self._service.search_conversations(role_id, query, max_items)

    # -- Knowledge Units ----------------------------------------------------

    def add_knowledge(
        self,
        role_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._service.add_knowledge(role_id, content, metadata)

    def search_knowledge(
        self,
        role_id: str,
        query: str,
        max_items: int = 10,
    ) -> List[Dict[str, Any]]:
        return self._service.search_knowledge(role_id, query, max_items)

    def archive_knowledge(
        self,
        role_id: str,
        knowledge_units: List[Dict[str, Any]],
    ) -> str:
        return self._service.archive_knowledge(role_id, knowledge_units)

    # -- Health / Capabilities ----------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        try:
            return {
                "status": "ok",
                "details": {
                    "provider": self.PROVIDER_NAME,
                    "version": self.PROVIDER_VERSION,
                    "data_dir": self._service.data_dir,
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "details": {"error": str(e)},
            }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.PROVIDER_NAME,
            "supports_embeddings": True,
            "supports_archive": True,
            "supports_conversation_search": True,
            "embedding_backend": getattr(
                self._service, "_embedding_backend", "unknown"
            ),
        }
