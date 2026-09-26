"""Local embeddings. Default is BAAI/bge-small-en-v1.5 - free, runs on CPU.

A deterministic hashing embedder (EMBED_BACKEND=hash) keeps the pipeline
runnable with no model download, which is what the unit tests use.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Sequence

from .config import Settings

#: BGE retrieval models expect this instruction on the *query* side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_WORD_RE = re.compile(r"[a-z0-9]+")


class Embedder:
    name: str = "embedder"
    dim: int = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """sentence-transformers wrapper; the model is loaded on first use."""

    def __init__(self, model_name: str, batch_size: int = 32, device: str | None = None) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._use_query_prefix = "bge" in model_name.lower() and "-en" in model_name.lower()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self._device)
            # renamed in sentence-transformers 6.x
            getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension"
            )
            self.dim = int(getter())
        return self._model

    def _encode(self, texts: Sequence[str], show_progress: bool = False) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,  # unit vectors -> cosine == dot product
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts, show_progress=len(texts) > 256)

    def embed_query(self, text: str) -> list[float]:
        payload = BGE_QUERY_PREFIX + text if self._use_query_prefix else text
        return self._encode([payload])[0]


class HashEmbedder(Embedder):
    """Offline bag-of-ngrams hashing. Lexical, but deterministic and instant."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = _WORD_RE.findall(text.lower())
        grams: Iterable[str] = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def get_embedder(settings: Settings) -> Embedder:
    if settings.embed_backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(settings.embed_model, batch_size=settings.embed_batch_size)
