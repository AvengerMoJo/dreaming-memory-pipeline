"""
Local filesystem storage backend for mojo_memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from mojo_memory.storage.base import StorageBackend


class LocalFileStorageBackend(StorageBackend):
    """Store JSON blobs under a base directory using relative keys."""

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        key = key.lstrip("/")
        return self.base_path / key

    def read_json(self, key: str) -> Any | None:
        path = self._resolve(key)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, key: str, data: Any) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str) -> bool:
        path = self._resolve(key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_keys(self, prefix: str = "") -> List[str]:
        keys: List[str] = []
        for path in self.base_path.rglob("*.json"):
            rel = path.relative_to(self.base_path).as_posix()
            if not prefix or rel.startswith(prefix):
                keys.append(rel)
        keys.sort()
        return keys

    def health_check(self) -> Dict[str, Any]:
        writable = True
        error = None
        probe = self.base_path / ".healthcheck.tmp"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - best effort
            writable = False
            error = str(exc)
        return {
            "ok": writable,
            "backend": "local_fs",
            "base_path": str(self.base_path),
            "error": error,
        }
