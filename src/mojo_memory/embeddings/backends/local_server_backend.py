"""HTTP embedding backend (legacy local + OpenAI-compatible)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from mojo_memory.embeddings.backends.base import EmbeddingBackend
from mojo_memory.embeddings.backends.random_backend import RandomBackend


class LocalServerBackend(EmbeddingBackend):
    """POST-based embedding backend.

    request_format:
      - "legacy" (default): body {"text": text} — old local-server contract
      - "openai":           body {"input": [...], "model": "...", "dimensions": N}
                            with `Authorization: Bearer <api_key>` header.
                            Targets OpenAI / OpenRouter / any OpenAI-compatible API.
    """

    def __init__(
        self,
        model_name: str = "local/server",
        embedding_dim: int = 768,
        server_url: str = "http://localhost:8080/embed",
        timeout_s: int = 10,
        request_format: str = "legacy",
        api_key: str = "",
        api_key_env: str = "",
        dimensions: Optional[int] = None,
        **_: Any,
    ):
        if request_format not in ("legacy", "openai"):
            raise ValueError(
                f"request_format must be 'legacy' or 'openai', got {request_format!r}"
            )
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.server_url = server_url
        self.timeout_s = timeout_s
        self.request_format = request_format
        self.dimensions = dimensions
        self.api_key = self._resolve_api_key(api_key, api_key_env)
        self._fallback = RandomBackend(model_name=model_name, embedding_dim=embedding_dim)

    @staticmethod
    def _resolve_api_key(api_key: str, api_key_env: str) -> str:
        if api_key:
            return api_key
        if api_key_env:
            return os.getenv(api_key_env, "") or ""
        return ""

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.request_format == "openai" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _body_single(self, text: str) -> Dict[str, Any]:
        if self.request_format == "openai":
            body: Dict[str, Any] = {"input": text, "model": self.model_name}
            if self.dimensions is not None:
                body["dimensions"] = self.dimensions
            return body
        return {"text": text}

    def _body_batch(self, texts: List[str]) -> Dict[str, Any]:
        if self.request_format == "openai":
            body: Dict[str, Any] = {"input": texts, "model": self.model_name}
            if self.dimensions is not None:
                body["dimensions"] = self.dimensions
            return body
        return {"texts": texts}

    def get_text_embedding(self, text: str, prompt_name: str = "passage") -> List[float]:
        del prompt_name
        try:
            res = requests.post(
                self.server_url,
                json=self._body_single(text),
                headers=self._headers(),
                timeout=self.timeout_s,
            )
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
            res = requests.post(
                self.server_url,
                json=self._body_batch(texts),
                headers=self._headers(),
                timeout=self.timeout_s,
            )
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
        # Surface "api" as the backend label when in OpenAI-compat mode so
        # stats/observability reflect the real protocol, not the legacy key.
        reported_backend = "api" if self.request_format == "openai" else "local"
        return {
            "backend": reported_backend,
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "server_url": self.server_url,
            "request_format": self.request_format,
        }

    def change_model(self, model_name: str) -> bool:
        self.model_name = model_name
        return True
