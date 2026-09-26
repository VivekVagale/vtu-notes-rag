"""Tiny JSON manifest so already-indexed PDFs are skipped on re-ingest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(path=path, entries=dict(data.get("files", {})))
            except (json.JSONDecodeError, OSError):
                pass  # corrupt manifest -> rebuild from scratch
        return cls(path=path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "files": self.entries}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def status(self, source: str, file_hash: str) -> str:
        """'new' | 'unchanged' | 'changed'"""
        entry = self.entries.get(source)
        if entry is None:
            return "new"
        return "unchanged" if entry.get("hash") == file_hash else "changed"

    def record(self, source: str, *, file_hash: str, subject: str, pages: int, chunks: int) -> None:
        self.entries[source] = {
            "hash": file_hash,
            "subject": subject,
            "pages": pages,
            "chunks": chunks,
            "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def forget(self, source: str) -> None:
        self.entries.pop(source, None)

    def prune(self, known_sources: set[str]) -> list[str]:
        """Drop entries whose PDF no longer exists. Returns removed sources."""
        gone = [s for s in self.entries if s not in known_sources]
        for source in gone:
            del self.entries[source]
        return gone
