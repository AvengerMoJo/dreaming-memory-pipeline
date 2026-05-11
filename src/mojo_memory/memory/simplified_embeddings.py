"""
Enhanced Embeddings Interface for MoJoAssistant.

This wrapper provides caching + similarity helpers while delegating embedding
operations to pluggable backends from mojo_memory.embeddings.registry.
"""

from __future__ import annotations

from typing import List, Dict, Any
import os
import json
import hashlib
import math

try:
    import numpy as np
    _numpy_available = True
except ImportError:
    np = None  # type: ignore[assignment]
    _numpy_available = False

from app.config.logging_config import get_logger
from mojo_memory.embeddings.registry import create_backend


class SimpleEmbedding:
    """Caching wrapper around a pluggable embedding backend."""

    def __init__(
        self,
        backend: str = "huggingface",
        model_name: str = "BAAI/bge-m3",
        api_key: str | None = None,
        server_url: str = "http://localhost:8080/embed",
        embedding_dim: int = 768,
        cache_dir: str = ".embedding_cache",
        device: str | None = None,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.backend = backend
        self.model_name = model_name
        self.api_key = api_key
        self.server_url = server_url
        self.embedding_dim: int | None = embedding_dim
        self.device = device

        self.model_version = f"{backend}:{model_name}:{embedding_dim}"
        self.metadata = {
            "model_version": self.model_version,
            "model_name": model_name,
            "backend": backend,
            "embedding_dim": embedding_dim,
            "created_at": None,
        }

        self.cache_dir = cache_dir
        self.cache: Dict[str, List[float]] = {}
        self._init_cache()

        self._backend_impl = create_backend(
            backend,
            model_name=model_name,
            embedding_dim=embedding_dim,
            server_url=server_url,
            api_key=api_key,
            device=device,
        )
        self._sync_from_backend_info()

    def _sync_from_backend_info(self) -> None:
        info = self._backend_impl.get_info()
        self.backend = info.get("backend", self.backend)
        self.model_name = info.get("model_name", self.model_name)
        self.embedding_dim = info.get("embedding_dim", self.embedding_dim)

    def _init_cache(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(
            self.cache_dir, f"{self.model_name.replace('/', '_')}_cache.json"
        )
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                self.logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                self.logger.error(f"Error loading embedding cache: {e}")

    def _save_cache(self) -> None:
        cache_file = os.path.join(
            self.cache_dir, f"{self.model_name.replace('/', '_')}_cache.json"
        )
        try:
            if 0 < len(self.cache) < 10000:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f)
        except Exception as e:
            self.logger.error(f"Error saving embedding cache: {e}")

    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        if _numpy_available:
            va = np.array(vec_a, dtype=np.float32)
            vb = np.array(vec_b, dtype=np.float32)
            dot = float(np.dot(va, vb))
            na = float(np.linalg.norm(va))
            nb = float(np.linalg.norm(vb))
        else:
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            na = math.sqrt(sum(x * x for x in vec_a))
            nb = math.sqrt(sum(x * x for x in vec_b))
        if na == 0 or nb == 0:
            return 0.0
        return float(dot / (na * nb))

    def get_text_embedding(self, text: str, prompt_name: str = "passage") -> List[float]:
        key = self._get_cache_key(text)
        if key in self.cache:
            return self.cache[key]
        emb = self._backend_impl.get_text_embedding(text, prompt_name=prompt_name)
        self.cache[key] = emb
        if len(self.cache) % 100 == 0:
            self._save_cache()
        return emb

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float] | None]:
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []
        results: List[List[float] | None] = [None] * len(texts)

        for i, text in enumerate(texts):
            key = self._get_cache_key(text)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            embeddings = self._backend_impl.get_batch_embeddings(uncached_texts)
            for i, emb in zip(uncached_indices, embeddings):
                key = self._get_cache_key(texts[i])
                self.cache[key] = emb
                results[i] = emb
            if len(uncached_texts) > 50:
                self._save_cache()

        return results

    def get_model_info(self) -> Dict[str, Any]:
        info = self._backend_impl.get_info()
        return {
            "backend": info.get("backend", self.backend),
            "model_name": info.get("model_name", self.model_name),
            "embedding_dim": info.get("embedding_dim", self.embedding_dim),
            "cache_size": len(self.cache),
            "device": info.get("device", self.device),
        }

    def change_model(self, model_name: str, backend: str | None = None) -> bool:
        try:
            old = self.model_name
            if backend and backend != self.backend:
                self.backend = backend
                self._backend_impl = create_backend(
                    backend,
                    model_name=model_name,
                    embedding_dim=int(self.embedding_dim or 768),
                    server_url=self.server_url,
                    api_key=self.api_key,
                    device=self.device,
                )
            else:
                self._backend_impl.change_model(model_name)

            self.model_name = model_name
            self._sync_from_backend_info()
            self._init_cache()
            self.logger.info(f"Changed embedding model from {old} to {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error changing embedding model: {e}")
            return False
