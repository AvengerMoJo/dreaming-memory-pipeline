"""Strategy registry — maps config names to RetrievalStrategy instances."""

from __future__ import annotations

from typing import Dict, Optional

from app.services.provider_contracts import RetrievalStrategy

_REGISTRY: Dict[str, RetrievalStrategy] = {}


def register_strategy(name: str, strategy: RetrievalStrategy) -> None:
    _REGISTRY[name] = strategy


def get_strategy(name: str = "semantic") -> Optional[RetrievalStrategy]:
    """Return a registered strategy by name, or None if unknown."""
    return _REGISTRY.get(name)


def _bootstrap() -> None:
    from mojo_memory.retrieval.semantic import SemanticStrategy
    from mojo_memory.retrieval.hybrid import HybridStrategy
    register_strategy("semantic", SemanticStrategy())
    register_strategy("hybrid", HybridStrategy())


_bootstrap()
