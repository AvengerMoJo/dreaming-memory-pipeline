"""
Mirror storage backend: read from primary, write to primary + mirrors.
"""

from __future__ import annotations

from typing import Any, Dict, List

from mojo_memory.storage.base import StorageBackend


class MirrorStorageBackend(StorageBackend):
    def __init__(
        self,
        primary: StorageBackend,
        mirrors: List[StorageBackend] | None = None,
        compare_on_read: bool = False,
    ):
        self.primary = primary
        self.mirrors = mirrors or []
        self.compare_on_read = compare_on_read

    def read_json(self, key: str) -> Any | None:
        primary_value = self.primary.read_json(key)
        if self.compare_on_read:
            for mirror in self.mirrors:
                try:
                    _ = mirror.read_json(key)
                except Exception:
                    pass
        return primary_value

    def write_json(self, key: str, data: Any) -> None:
        self.primary.write_json(key, data)
        for mirror in self.mirrors:
            try:
                mirror.write_json(key, data)
            except Exception:
                # Mirror failures are tolerated; primary defines success path.
                pass

    def exists(self, key: str) -> bool:
        return self.primary.exists(key)

    def delete(self, key: str) -> bool:
        deleted = self.primary.delete(key)
        for mirror in self.mirrors:
            try:
                mirror.delete(key)
            except Exception:
                pass
        return deleted

    def list_keys(self, prefix: str = "") -> List[str]:
        return self.primary.list_keys(prefix)

    def health_check(self) -> Dict[str, Any]:
        primary_health = self.primary.health_check()
        mirror_health = [m.health_check() for m in self.mirrors]
        return {
            "ok": bool(primary_health.get("ok")),
            "backend": "mirror",
            "primary": primary_health,
            "mirrors": mirror_health,
        }
