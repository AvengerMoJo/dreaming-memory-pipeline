"""
DuckDB storage backend for mojo_memory JSON blobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from mojo_memory.storage.base import StorageBackend


class DuckDBStorageBackend(StorageBackend):
    """Store JSON payloads in a DuckDB key-value table."""

    def __init__(self, db_path: str | Path):
        try:
            import duckdb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("duckdb is required for DuckDBStorageBackend") from exc
        self._duckdb = duckdb
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mojo_storage (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def read_json(self, key: str) -> Any | None:
        row = self.conn.execute(
            "SELECT value_json FROM mojo_storage WHERE key = ?", [key]
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def write_json(self, key: str, data: Any) -> None:
        payload = json.dumps(data, ensure_ascii=False)
        self.conn.execute(
            """
            INSERT INTO mojo_storage (key, value_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            [key, payload],
        )

    def exists(self, key: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM mojo_storage WHERE key = ? LIMIT 1", [key]
        ).fetchone()
        return row is not None

    def delete(self, key: str) -> bool:
        before = self.conn.execute(
            "SELECT COUNT(*) FROM mojo_storage WHERE key = ?", [key]
        ).fetchone()[0]
        if before == 0:
            return False
        self.conn.execute("DELETE FROM mojo_storage WHERE key = ?", [key])
        return True

    def list_keys(self, prefix: str = "") -> List[str]:
        if prefix:
            rows = self.conn.execute(
                "SELECT key FROM mojo_storage WHERE key LIKE ? ORDER BY key ASC",
                [f"{prefix}%"],
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT key FROM mojo_storage ORDER BY key ASC"
            ).fetchall()
        return [row[0] for row in rows]

    def health_check(self) -> Dict[str, Any]:
        ok = True
        error = None
        try:
            self.conn.execute("SELECT 1").fetchone()
        except Exception as exc:  # pragma: no cover
            ok = False
            error = str(exc)
        return {
            "ok": ok,
            "backend": "duckdb",
            "db_path": self.db_path,
            "error": error,
        }
