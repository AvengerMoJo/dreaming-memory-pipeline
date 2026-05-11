"""HybridStrategy — multi-model search with priority fallback and score fusion."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from app.services.provider_contracts import RetrievalStrategy, ScoredResult


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class HybridStrategy(RetrievalStrategy):
    """
    Rank candidates across multiple embedding models with priority fallback.

    For each candidate, scores are computed for every model present in
    ``model_priority`` and blended via weighted average.  Models earlier in
    the priority list carry more weight.  This mirrors the previous hardcoded
    behaviour in HybridMemoryService._get_multi_model_context but makes it
    configurable without code changes.
    """

    DEFAULT_PRIORITY = ["bge-m3:1024", "gemma:768", "gemma:256"]

    def __init__(
        self,
        model_priority: Optional[List[str]] = None,
    ) -> None:
        self._priority = model_priority or self.DEFAULT_PRIORITY

    @property
    def name(self) -> str:
        return "hybrid"

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
            available = [m for m in self._priority if m in embeddings]
            if not available:
                # Fall back to any available model
                available = list(embeddings.keys())
            if not available:
                continue

            # Weighted average: earlier models get higher weight
            total_weight = 0.0
            weighted_score = 0.0
            for rank, model_key in enumerate(available):
                weight = 1.0 / (rank + 1)
                score = _cosine(query_embedding, embeddings[model_key])
                weighted_score += weight * score
                total_weight += weight

            final_score = weighted_score / total_weight if total_weight > 0 else 0.0

            if final_score >= threshold:
                results.append(
                    ScoredResult(
                        content=candidate.get("text_content", ""),
                        score=final_score,
                        source=candidate.get("source", "unknown"),
                        metadata=candidate.get("metadata", candidate.get("user_metadata", {})),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]
