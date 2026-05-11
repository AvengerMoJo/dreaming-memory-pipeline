"""Base contract for mojo_memory embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class EmbeddingBackend(ABC):
    @abstractmethod
    def get_text_embedding(self, text: str, prompt_name: str = "passage") -> List[float]:
        ...

    @abstractmethod
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def change_model(self, model_name: str) -> bool:
        ...
