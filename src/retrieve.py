"""Retrieval: embed the question, search Chroma, rerank, return cited sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .rerank import Reranker, dedupe_by_page, get_reranker
from .store import PUBLIC_TIERS, VectorStore, build_scope_where


@dataclass(frozen=True)
class Source:
    """One retrieved chunk, already carrying everything a citation needs."""

    text: str
    file: str
    page: int
    subject: str
    source: str
    chunk_id: str
    visibility: str
    score: float
    rerank_score: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"[{self.file}, p.{self.page}]"

    def snippet(self, limit: int = 400) -> str:
        body = " ".join(self.text.split())
        return body if len(body) <= limit else body[: limit - 3].rstrip() + "..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "page": self.page,
            "subject": self.subject,
            "source": self.source,
            "visibility": self.visibility,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "citation": self.citation,
            "text": self.text,
        }


def hit_to_source(hit: dict[str, Any]) -> Source:
    meta = hit.get("metadata", {}) or {}
    return Source(
        text=hit.get("text", ""),
        file=str(meta.get("file", "unknown.pdf")),
        page=int(meta.get("page", 0) or 0),
        subject=str(meta.get("subject", "General")),
        source=str(meta.get("source", meta.get("file", "unknown.pdf"))),
        chunk_id=str(hit.get("id", "")),
        visibility=str(meta.get("visibility", "curated")),
        score=float(hit.get("score", 0.0)),
        rerank_score=float(hit.get("rerank_score", hit.get("score", 0.0))),
        extra={"lexical_score": hit.get("lexical_score")},
    )


class Retriever:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or VectorStore(self.settings)
        self.embedder = embedder or get_embedder(self.settings)
        self.reranker = reranker if reranker is not None else get_reranker(self.settings)

    def retrieve(
        self,
        question: str,
        *,
        k: int | None = None,
        subject: str | None = None,
        scopes: Sequence[str] | None = None,
        owner_id: str | None = None,
        fetch_k: int | None = None,
        keep_per_page: int = 2,
    ) -> list[Source]:
        """Search the given library tiers.

        scopes=None falls back to the public tiers - never to an unfiltered read.
        scopes=[] means the caller is entitled to nothing, so nothing comes back.
        """
        question = (question or "").strip()
        if not question:
            return []
        if scopes is not None and not list(scopes):
            return []
        top_k = k or self.settings.top_k
        candidates = fetch_k or max(self.settings.fetch_k, top_k)
        if self.reranker.name == "none":
            candidates = top_k

        where = build_scope_where(
            subject=subject if subject and subject != "All" else None,
            tiers=list(scopes) if scopes else list(PUBLIC_TIERS),
            owner_id=owner_id,
        )
        hits = self.store.query(self.embedder.embed_query(question), candidates, where)
        if not hits:
            return []
        hits = self.reranker.rerank(question, hits, len(hits))
        hits = dedupe_by_page(hits, keep_per_page=keep_per_page)[:top_k]
        return [hit_to_source(h) for h in hits]

    # --- convenience for the UIs ---------------------------------------
    def subjects(self) -> list[str]:
        return self.store.subjects()

    def count(self) -> int:
        return self.store.count()
