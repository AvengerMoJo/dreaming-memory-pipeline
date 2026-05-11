"""HuggingFace sentence-transformers embedding backend."""

from __future__ import annotations

from typing import Any, Dict, List

from mojo_memory.embeddings.backends.base import EmbeddingBackend
from mojo_memory.embeddings.backends.random_backend import RandomBackend

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


class HuggingFaceBackend(EmbeddingBackend):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        embedding_dim: int = 768,
        device: str | None = None,
        **_: Any,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.model = None
        self._fallback = RandomBackend(model_name=model_name, embedding_dim=embedding_dim)
        self._load_model()

    def _load_model(self) -> None:
        if SentenceTransformer is None:
            return
        try:
            self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
            if self.device and self.model is not None:
                self.model.to(self.device)
            if self.model is not None:
                dim = self.model.get_sentence_embedding_dimension()
                if dim:
                    self.embedding_dim = int(dim)
        except Exception:
            self.model = None

    def _to_list(self, emb: Any) -> List[float]:
        if np is not None and isinstance(emb, np.ndarray):
            return emb.astype(np.float32).tolist()
        return [float(x) for x in emb]

    def get_text_embedding(self, text: str, prompt_name: str = "passage") -> List[float]:
        if self.model is None:
            return self._fallback.get_text_embedding(text, prompt_name=prompt_name)
        try:
            if self.model_name == "nomic-ai/nomic-embed-text-v2-moe":
                emb = self.model.encode(text, prompt_name=prompt_name)
            else:
                emb = self.model.encode(text)
            return self._to_list(emb)
        except Exception:
            return self._fallback.get_text_embedding(text, prompt_name=prompt_name)

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.model is None:
            return [self._fallback.get_text_embedding(t) for t in texts]
        try:
            embs = self.model.encode(texts)
            if np is not None and isinstance(embs, np.ndarray):
                return embs.astype(np.float32).tolist()
            return [list(e) for e in embs]
        except Exception:
            return [self._fallback.get_text_embedding(t) for t in texts]

    def get_info(self) -> Dict[str, Any]:
        return {
            "backend": "huggingface",
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "model_loaded": self.model is not None,
        }

    def change_model(self, model_name: str) -> bool:
        self.model_name = model_name
        self._load_model()
        return True
