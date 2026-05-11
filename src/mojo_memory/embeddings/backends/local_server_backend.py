"""HTTP local embedding backend."""

from __future__ import annotations

from typing import Any, Dict, List

import requests

from mojo_memory.embeddings.backends.base import EmbeddingBackend
from mojo_memory.embeddings.backends.random_backend import RandomBackend


class LocalServerBackend(EmbeddingBackend):
    def __init__(
        self,
        model_name: str = "local/server",
        embedding_dim: int = 768,
        server_url: str = "http://localhost:8080/embed",
        timeout_s: int = 10,
        **_: Any,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.server_url = server_url
        self.timeout_s = timeout_s
        self._fallback = RandomBackend(model_name=model_name, embedding_dim=embedding_dim)

    def get_text_embedding(self, text: str, prompt_name: str = "passage") -> List[float]:
        del prompt_name
        try:
            res = requests.post(self.server_url, json={"text": text}, timeout=self.timeout_s)
            if res.status_code == 200:
                payload = res.json()
                if "embedding" in payload:
                    return payload["embedding"]
                if "data" in payload and payload["data"]:
                    return payload["data"][0]["embedding"]
        except Exception:
            pass
        return self._fallback.get_text_embedding(text)

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            res = requests.post(self.server_url, json={"texts": texts}, timeout=self.timeout_s)
            if res.status_code == 200:
                payload = res.json()
                if "embeddings" in payload:
                    return payload["embeddings"]
                if "data" in payload:
                    return [item["embedding"] for item in payload["data"]]
        except Exception:
            pass
        return [self.get_text_embedding(t) for t in texts]

    def get_info(self) -> Dict[str, Any]:
        return {
            "backend": "local",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "server_url": self.server_url,
        }

    def change_model(self, model_name: str) -> bool:
        self.model_name = model_name
        return True
