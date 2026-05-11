"""SemanticStrategy — cosine similarity over a single embedding model."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from app.services.provider_contracts import RetrievalStrategy, ScoredResult


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticStrategy(RetrievalStrategy):
    """
    Rank candidates by cosine similarity using one embedding model.

    Picks the first model_key found in each candidate's ``embeddings`` dict
    that matches ``preferred_model``, or falls back to the first available key.
    """

    def __init__(self, preferred_model: str = "bge-m3:1024") -> None:
        self._preferred_model = preferred_model

    @property
    def name(self) -> str:
        return "semantic"

    def search(
        self,
        query_embedding: List[float],
        candidates: List[Dict[str, Any]],
        *,
        max_results: int = 10,
        threshold: float = 0.3,
    ) -> List[ScoredResult]:
        results: List[ScoredResult] = []
        for candidate in candidates:
            embeddings: Dict[str, List[float]] = candidate.get("embeddings", {})
            # Prefer configured model, fall back to first available
            model_key = (
                self._preferred_model
                if self._preferred_model in embeddings
                else next(iter(embeddings), None)
            )
            if model_key is None:
                continue
            score = _cosine(query_embedding, embeddings[model_key])
            if score >= threshold:
                results.append(
                    ScoredResult(
                        content=candidate.get("text_content", ""),
                        score=score,
                        source=candidate.get("source", "unknown"),
                        metadata=candidate.get("metadata", candidate.get("user_metadata", {})),
                    )
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]
