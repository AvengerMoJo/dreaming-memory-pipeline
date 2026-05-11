"""
AWS S3 storage backend for mojo_memory JSON blobs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from mojo_memory.storage.base import StorageBackend


class S3StorageBackend(StorageBackend):
    """
    Key-value JSON storage backed by S3.

    Requires boto3 at runtime:
      pip install boto3
    """

    def __init__(self, bucket: str, prefix: str = "", s3_client: Any | None = None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if s3_client is not None:
            self.s3 = s3_client
            return
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("boto3 is required for S3StorageBackend") from exc
        self.s3 = boto3.client("s3")

    def _full_key(self, key: str) -> str:
        clean = key.lstrip("/")
        if not self.prefix:
            return clean
        return f"{self.prefix}/{clean}"

    def read_json(self, key: str) -> Any | None:
        full_key = self._full_key(key)
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=full_key)
        except Exception:
            return None
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)

    def write_json(self, key: str, data: Any) -> None:
        full_key = self._full_key(key)
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket,
            Key=full_key,
            Body=payload,
            ContentType="application/json",
        )

    def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            self.s3.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        full_key = self._full_key(key)
        if not self.exists(key):
            return False
        self.s3.delete_object(Bucket=self.bucket, Key=full_key)
        return True

    def list_keys(self, prefix: str = "") -> List[str]:
        effective_prefix = self._full_key(prefix) if prefix else (self.prefix + "/" if self.prefix else "")
        paginator = self.s3.get_paginator("list_objects_v2")
        keys: List[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=effective_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.prefix and key.startswith(self.prefix + "/"):
                    key = key[len(self.prefix) + 1 :]
                keys.append(key)
        keys.sort()
        return keys

    def health_check(self) -> Dict[str, Any]:
        ok = True
        error = None
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception as exc:
            ok = False
            error = str(exc)
        return {
            "ok": ok,
            "backend": "s3",
            "bucket": self.bucket,
            "prefix": self.prefix,
            "error": error,
        }
