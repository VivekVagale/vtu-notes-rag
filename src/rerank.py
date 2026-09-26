"""Reranking of the candidate set returned by vector search.

lexical (default): no extra model, blends the embedding score with query-term
coverage, which fixes most "right topic, wrong page" cases.
cross-encoder: heavier and better, used when RERANKER=cross-encoder.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from .config import Settings

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "what", "which", "how", "why", "with", "explain", "define", "describe",
    "list", "state", "give", "write", "note", "short", "briefly", "it", "its",
    "that", "this", "does", "do", "be", "by", "from", "as", "at", "you",
}


#: Contributed material is searchable but never outranks vetted notes on a tie.
TRUST = {"curated": 1.0, "private": 1.0, "community": 0.94}


def trust_of(hit: dict[str, Any]) -> float:
    return TRUST.get(str((hit.get("metadata") or {}).get("visibility", "curated")), 0.9)


def _terms(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2]


class Reranker:
    """No-op reranker: keeps pure vector order."""

    name = "none"

    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        for hit in hits:
            hit.setdefault("rerank_score", hit.get("score", 0.0))
        return hits[:top_k]


class LexicalReranker(Reranker):
    name = "lexical"

    def __init__(self, vector_weight: float = 0.75) -> None:
        self.vector_weight = vector_weight

    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        q_set = set(_terms(query))
        phrase = " ".join(_WORD_RE.findall(query.lower()))
        for hit in hits:
            body = hit.get("text", "").lower()
            doc_terms = set(_WORD_RE.findall(body))
            coverage = len(q_set & doc_terms) / len(q_set) if q_set else 0.0
            bonus = 0.05 if phrase and phrase in body else 0.0
            vec = float(hit.get("score", 0.0))
            hit["lexical_score"] = coverage
            blended = self.vector_weight * vec + (1.0 - self.vector_weight) * coverage + bonus
            hit["rerank_score"] = min(1.0, blended) * trust_of(hit)
        hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        return hits[:top_k]


class CrossEncoderReranker(Reranker):
    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._fallback = LexicalReranker()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not hits:
            return []
        try:
            scores = self.model.predict([(query, h.get("text", "")) for h in hits])
        except Exception:  # model missing / offline -> degrade, never crash a query
            return self._fallback.rerank(query, hits, top_k)
        for hit, raw in zip(hits, scores):
            hit["rerank_score"] = (1.0 / (1.0 + math.exp(-float(raw)))) * trust_of(hit)
        hits.sort(key=lambda h: h["rerank_score"], reverse=True)
        return hits[:top_k]


def get_reranker(settings: Settings) -> Reranker:
    choice = (settings.reranker or "none").lower()
    if choice in {"cross-encoder", "cross_encoder", "ce"}:
        return CrossEncoderReranker(settings.cross_encoder_model)
    if choice == "lexical":
        return LexicalReranker()
    return Reranker()


def dedupe_by_page(hits: Sequence[dict[str, Any]], keep_per_page: int = 2) -> list[dict[str, Any]]:
    """Stop one dense page from eating the whole context window."""
    counts: dict[tuple[str, int], int] = {}
    kept: list[dict[str, Any]] = []
    for hit in hits:
        meta = hit.get("metadata", {})
        key = (str(meta.get("source", "")), int(meta.get("page", 0) or 0))
        if counts.get(key, 0) >= keep_per_page:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(hit)
    return kept
