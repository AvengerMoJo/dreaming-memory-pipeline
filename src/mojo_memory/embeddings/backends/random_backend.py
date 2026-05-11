"""Deterministic random embedding backend for tests/fallback."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List

from mojo_memory.embeddings.backends.base import EmbeddingBackend


class RandomBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "random/default", embedding_dim: int = 768, **_: Any):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    def get_text_embedding(self, text: str, prompt_name: str = "passage") -> List[float]:
        del prompt_name
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        random.seed(seed)
        vec = [random.gauss(0, 1) for _ in range(self.embedding_dim)]
        mag = math.sqrt(sum(x * x for x in vec))
        return [x / mag for x in vec] if mag > 0 else vec

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.get_text_embedding(t) for t in texts]

    def get_info(self) -> Dict[str, Any]:
        return {
            "backend": "random",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
        }

    def change_model(self, model_name: str) -> bool:
        self.model_name = model_name
        return True
