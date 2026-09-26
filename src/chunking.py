"""Token-aware chunking.

Uses the embedding model's own tokenizer when transformers is available so
"~500 tokens" means real model tokens. Falls back to a word tokenizer with a
words->tokens ratio so the module still works offline (and in unit tests).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

_WORD_RE = re.compile(r"\S+")

#: English subword tokenizers emit roughly 1.3 tokens per whitespace word.
WORDS_PER_TOKEN = 1.3


class BaseTokenizer:
    name = "base"

    def spans(self, text: str) -> list[tuple[int, int]]:
        raise NotImplementedError

    def budget(self, n_tokens: int) -> int:
        """How many of *my* units correspond to n model tokens."""
        return max(1, n_tokens)

    def count(self, text: str) -> int:
        return len(self.spans(text))


class WordTokenizer(BaseTokenizer):
    name = "word"

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]

    def budget(self, n_tokens: int) -> int:
        return max(1, int(n_tokens / WORDS_PER_TOKEN))


class HFTokenizer(BaseTokenizer):
    """Real subword tokenizer with character offsets (fast tokenizers only)."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self._tok.is_fast:  # pragma: no cover - offsets need a fast tokenizer
            raise RuntimeError(f"{model_name} has no fast tokenizer")
        self.name = model_name

    def spans(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        # Drop zero-width offsets (special/continuation artefacts).
        return [(s, e) for s, e in enc["offset_mapping"] if e > s]


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str | None = None) -> BaseTokenizer:
    """Best available tokenizer; never raises."""
    if model_name:
        try:
            return HFTokenizer(model_name)
        except Exception:  # offline, no transformers, no cached files
            pass
    return WordTokenizer()


@dataclass(frozen=True)
class TextChunk:
    text: str
    index: int
    char_start: int
    char_end: int
    n_units: int


def chunk_text(
    text: str,
    *,
    max_tokens: int = 500,
    overlap: int = 80,
    tokenizer: BaseTokenizer | None = None,
    min_chars: int = 40,
) -> list[TextChunk]:
    """Split text into overlapping chunks of ~max_tokens model tokens."""
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    if not text or not text.strip():
        return []

    tok = tokenizer or WordTokenizer()
    spans = tok.spans(text)
    if not spans:
        return []

    window = tok.budget(max_tokens)
    step = max(1, window - tok.budget(overlap))

    chunks: list[TextChunk] = []
    for start_i in range(0, len(spans), step):
        piece = spans[start_i : start_i + window]
        if not piece:
            break
        raw = text[piece[0][0] : piece[-1][1]]
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        if len(body) >= min_chars or (not chunks and body):
            chunks.append(
                TextChunk(
                    text=body,
                    index=len(chunks),
                    char_start=piece[0][0] + lead,
                    char_end=piece[0][0] + lead + len(body),
                    n_units=len(piece),
                )
            )
        if start_i + window >= len(spans):
            break

    # A short tail window can be fully contained in its predecessor.
    if len(chunks) > 1 and chunks[-1].char_end <= chunks[-2].char_end:
        chunks.pop()
    return chunks
