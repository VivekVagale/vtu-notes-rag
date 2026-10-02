"""ChromaDB wrapper, persisted to disk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .config import Settings

_MAX_SCAN = 100_000


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    text: str
    metadata: dict[str, Any]


#: Tiers a caller with no identity may ever read.
PUBLIC_TIERS = ("curated", "community")
KNOWN_TIERS = ("curated", "community", "private")


def _tier_clause(tiers: Sequence[str]) -> dict[str, Any]:
    """Chroma rejects an empty $in and a one-element $and/$or, so never emit either."""
    tiers = list(tiers)
    return {"visibility": tiers[0]} if len(tiers) == 1 else {"visibility": {"$in": tiers}}


def build_scope_where(
    *,
    subject: str | None = None,
    tiers: Sequence[str],
    owner_id: str | None = None,
) -> dict[str, Any]:
    """Where clause for a tier-scoped read.

    Private chunks are reachable only through a positive owner predicate - never
    through a negative one, because Chroma's $ne matches rows that lack the key
    entirely (verified on 1.5.9), which would fail open straight into someone
    else's notes.
    """
    wanted = list(dict.fromkeys(tiers))
    if not wanted:
        raise ValueError("at least one tier is required; an empty scope must return no sources")
    unknown = [t for t in wanted if t not in KNOWN_TIERS]
    if unknown:
        raise ValueError(f"unknown tier(s): {', '.join(unknown)}")
    if "private" in wanted and not owner_id:
        raise ValueError("the private tier requires an owner predicate")

    branches: list[dict[str, Any]] = []
    shared = [t for t in wanted if t != "private"]
    if shared:
        branches.append(_tier_clause(shared))
    if "private" in wanted:
        branches.append({"$and": [{"visibility": "private"}, {"owner_id": owner_id}]})

    clause = branches[0] if len(branches) == 1 else {"$or": branches}
    if not subject:
        return clause
    return {"$and": [{"subject": subject}, clause]}


def build_where(
    subject: str | None = None,
    source: str | None = None,
    visibility: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if subject:
        clauses.append({"subject": subject})
    if source:
        clauses.append({"source": source})
    if visibility:
        tiers = list(visibility)
        clauses.append(
            {"visibility": tiers[0]} if len(tiers) == 1 else {"visibility": {"$in": tiers}}
        )
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class VectorStore:
    """Thin, dependency-lazy facade over a persistent Chroma collection."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.space = "cosine"
        self._client = None
        self._collection = None

    # --- lifecycle -----------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.settings.index_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.settings.index_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            try:
                self._collection = self.client.get_or_create_collection(
                    name=self.settings.collection,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                # Older/newer Chroma may reject the hint; vectors are L2-normalised
                # so the ranking is identical either way.
                self._collection = self.client.get_or_create_collection(
                    name=self.settings.collection
                )
        return self._collection

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.settings.collection)
        except Exception:
            pass
        self._collection = None

    # --- writes --------------------------------------------------------
    def add(
        self,
        records: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
        batch_size: int = 256,
    ) -> int:
        if not records:
            return 0
        if len(records) != len(embeddings):
            raise ValueError("records and embeddings length mismatch")
        for start in range(0, len(records), batch_size):
            window = records[start : start + batch_size]
            self.collection.upsert(
                ids=[r.id for r in window],
                documents=[r.text for r in window],
                metadatas=[r.metadata for r in window],
                embeddings=[list(e) for e in embeddings[start : start + batch_size]],
            )
        return len(records)

    def delete_by_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    # --- reads ---------------------------------------------------------
    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def query(
        self,
        embedding: Sequence[float],
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if k <= 0 or self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, chunk_id in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else 1.0
            hits.append(
                {
                    "id": chunk_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": dict(metas[i]) if i < len(metas) and metas[i] else {},
                    "distance": distance,
                    "score": self._similarity(distance),
                }
            )
        return hits

    def _similarity(self, distance: float) -> float:
        if self.space == "cosine":
            return max(0.0, min(1.0, 1.0 - distance))
        return 1.0 / (1.0 + max(0.0, distance))  # l2 fallback

    def _all_metadatas(self) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        got = self.collection.get(include=["metadatas"], limit=_MAX_SCAN)
        return [dict(m) for m in (got.get("metadatas") or []) if m]

    def subjects(self) -> list[str]:
        return sorted({str(m.get("subject", "General")) for m in self._all_metadatas()})

    def indexed_files(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for meta in self._all_metadatas():
            source = str(meta.get("source", meta.get("file", "?")))
            entry = seen.setdefault(
                source,
                {
                    "source": source,
                    "file": meta.get("file", source),
                    "subject": meta.get("subject", "General"),
                    "chunks": 0,
                    "pages": 0,
                },
            )
            entry["chunks"] += 1
            entry["pages"] = max(entry["pages"], int(meta.get("page", 0) or 0))
        return sorted(seen.values(), key=lambda e: (e["subject"], e["file"]))
