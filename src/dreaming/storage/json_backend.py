"""
JSON File Storage Backend

Stores archives as JSON files in ~/.memory/dreams/ (or configurable path).
Compatible with the original MoJoAssistant dreaming file layout.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from dreaming.storage.base import StorageBackend


class JsonFileBackend(StorageBackend):
    """
    JSON file-based storage backend.

    File layout:
        {storage_path}/{conversation_id}/archive_v{N}.json
        {storage_path}/{conversation_id}/manifest.json
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Args:
            storage_path: Root directory for archives.
                          Defaults to ~/.memory/dreams/
        """
        self.storage_path = storage_path or Path.home() / ".memory" / "dreams"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _archive_version_from_path(self, path: Path) -> Optional[int]:
        """Extract numeric version from archive filename like archive_v12.json."""
        match = re.match(r"archive_v(\d+)\.json$", path.name)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _get_archive_files_sorted(self, conv_dir: Path) -> List[Path]:
        """Get archive files sorted by numeric version ascending."""
        files = list(conv_dir.glob("archive_v*.json"))
        files_with_versions = []
        for f in files:
            version = self._archive_version_from_path(f)
            if version is not None:
                files_with_versions.append((version, f))
        files_with_versions.sort(key=lambda item: item[0])
        return [f for _, f in files_with_versions]

    def _manifest_path(self, conversation_id: str) -> Path:
        """Path to per-conversation manifest file."""
        return self.storage_path / conversation_id / "manifest.json"

    def _load_manifest(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load manifest if present."""
        manifest_path = self._manifest_path(conversation_id)
        if not manifest_path.exists():
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return None

    def _build_manifest_from_existing_archives(
        self, conversation_id: str
    ) -> Dict[str, Any]:
        """Bootstrap manifest from existing archive_v*.json files."""
        conv_dir = self.storage_path / conversation_id
        archive_files = self._get_archive_files_sorted(conv_dir)
        versions = [self._archive_version_from_path(p) for p in archive_files]
        versions = [v for v in versions if v is not None]
        latest = versions[-1] if versions else 0

        version_map: Dict[str, Any] = {}
        for v in versions:
            version_map[str(v)] = {
                "status": "active" if v == latest else "superseded",
                "storage_location": "hot" if v == latest else "cold",
                "is_latest": v == latest,
                "previous_version": (v - 1) if v > 1 else None,
                "supersedes_version": (v - 1) if v > 1 else None,
            }

        return {
            "conversation_id": conversation_id,
            "latest_version": latest,
            "updated_at": datetime.now().isoformat(),
            "versions": version_map,
        }

    def _get_or_init_manifest(
        self, conversation_id: str, persist_if_missing: bool = True
    ) -> Dict[str, Any]:
        """Get manifest, bootstrapping from existing archives if needed."""
        manifest = self._load_manifest(conversation_id)
        if manifest is not None:
            return manifest
        manifest = self._build_manifest_from_existing_archives(conversation_id)
        if persist_if_missing:
            self.update_manifest(conversation_id, manifest)
        return manifest

    # -- StorageBackend interface --

    def save_archive(self, conversation_id: str, version: int, data: dict) -> None:
        """Save archive to disk as JSON"""
        conv_dir = self.storage_path / conversation_id
        conv_dir.mkdir(parents=True, exist_ok=True)

        filename = f"archive_v{version}.json"
        archive_path = conv_dir / filename
        temp_path = conv_dir / f"{filename}.tmp"

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.replace(archive_path)

    def load_archive(self, conversation_id: str, version: int | None = None) -> dict | None:
        """Load archive from disk"""
        conv_dir = self.storage_path / conversation_id
        if not conv_dir.exists():
            return None

        if version is not None:
            archive_path = conv_dir / f"archive_v{version}.json"
            if not archive_path.exists():
                return None
        else:
            # Get latest version
            manifest = self._get_or_init_manifest(conversation_id)
            latest_version = int(manifest.get("latest_version", 0))
            if latest_version <= 0:
                archive_files = self._get_archive_files_sorted(conv_dir)
                if not archive_files:
                    return None
                archive_path = archive_files[-1]
            else:
                archive_path = conv_dir / f"archive_v{latest_version}.json"
                if not archive_path.exists():
                    archive_files = self._get_archive_files_sorted(conv_dir)
                    if not archive_files:
                        return None
                    archive_path = archive_files[-1]

        with open(archive_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_archives(self) -> list[dict]:
        """List all available archives"""
        archives = []

        for conv_dir in self.storage_path.iterdir():
            if not conv_dir.is_dir():
                continue

            archive_files = self._get_archive_files_sorted(conv_dir)
            if not archive_files:
                continue

            # Get latest version
            latest_path = archive_files[-1]
            latest_version = self._archive_version_from_path(latest_path)

            try:
                with open(latest_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            manifest = self._load_manifest(conv_dir.name) or {}
            latest_meta = (
                manifest.get("versions", {}).get(str(latest_version), {})
                if latest_version is not None
                else {}
            )

            archives.append({
                "conversation_id": conv_dir.name,
                "latest_version": latest_version if latest_version is not None else data.get("version", 1),
                "quality_level": data.get("quality_level", "unknown"),
                "created_at": data.get("created_at"),
                "status": latest_meta.get("status", "unknown"),
                "storage_location": latest_meta.get("storage_location", "unknown"),
                "entities_count": len(data.get("entities", [])),
                "chunks_count": len(data.get("b_chunks", [])),
                "clusters_count": len(data.get("c_clusters", []))
            })

        return archives

    def get_manifest(self, conversation_id: str) -> dict | None:
        """Get manifest, bootstrapping from archives if needed."""
        manifest = self._load_manifest(conversation_id)
        if manifest is not None:
            return manifest
        conv_dir = self.storage_path / conversation_id
        if not conv_dir.exists():
            return None
        return self._build_manifest_from_existing_archives(conversation_id)

    def update_manifest(self, conversation_id: str, manifest: dict) -> None:
        """Atomically save manifest."""
        conv_dir = self.storage_path / conversation_id
        conv_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self._manifest_path(conversation_id)
        temp_path = manifest_path.with_suffix(".json.tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        temp_path.replace(manifest_path)

    def get_latest_version(self, conversation_id: str) -> int:
        """Get latest archive version (0 if none exist)."""
        manifest = self._get_or_init_manifest(
            conversation_id, persist_if_missing=False
        )
        return int(manifest.get("latest_version", 0))
