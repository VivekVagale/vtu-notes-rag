"""Offline answer composer (LLM_PROVIDER=extractive).

No API key, no model server: the answer is stitched together from the sentences
of the retrieved passages, each carrying its own page citation. Deterministic,
which is exactly what the tests and a zero-setup demo need.
"""

from __future__ import annotations

import re
from typing import Sequence

from .config import NOT_FOUND_MESSAGE
from .rerank import _terms
from .retrieve import Source

# A period after a single digit is a list marker ("1. Mutual exclusion"), not an end.
_SENT_SPLIT = re.compile(r"(?<![0-9]\.)(?<=[.!?])\s+|\n+")
_NEW_BLOCK = re.compile(r"^([-*•]|\d+[.)]|[A-Z][A-Za-z ]{0,40}:)\s")
_DEFINITION_CUES = ("is defined as", "is a ", "is an ", "refers to", "means ", "is the ")
_DIAGRAM_CUES = ("diagram", "figure", "fig.", "table", "graph", "tree", "chart", "state")
_TOC_LINE = re.compile(r"^(page|contents|unit|module|chapter)\s*\d*\s*[-:.]?", re.IGNORECASE)
_SECTION_HEADING = re.compile(r"^\d+(\.\d+)+\s+[A-Z]")
_MARKS_LINE = re.compile(r"\(\d+\s*marks?\)", re.IGNORECASE)
# "3. Explain 3NF and BCNF with an example." is an exercise from the notes, not a fact.
_EXERCISE = re.compile(
    r"^\d+[.)]\s+(explain|define|state|list|write|discuss|derive|describe|prove"
    r"|compare|differentiate|give|draw|illustrate)\b",
    re.IGNORECASE,
)
_MIN_SENTENCE_CHARS = 30
_MAX_HEADING_CHARS = 90


def _is_heading(line: str) -> bool:
    """Headings and contents lines make useless answer bullets."""
    if len(line) < _MAX_HEADING_CHARS and not line.endswith("."):
        return True
    if _MARKS_LINE.search(line) or _EXERCISE.match(line):
        return True
    return bool(_TOC_LINE.match(line)) or bool(_SECTION_HEADING.match(line))


def _unwrap(text: str) -> str:
    """Undo the hard wrapping of PDF text without gluing headings to paragraphs."""
    lines: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        previous = lines[-1] if lines else ""
        continues = (
            previous
            and len(previous) > 60
            and not previous.endswith((".", "!", "?", ":"))
            and line
            and not _NEW_BLOCK.match(line)
        )
        if continues:
            lines[-1] = f"{previous} {line}"
        else:
            lines.append(line)
    return "\n".join(lines)


def sentences(text: str) -> list[str]:
    parts = [" ".join(raw.split()) for raw in _SENT_SPLIT.split(_unwrap(text))]
    merged: list[str] = []
    for part in parts:
        if not part:
            continue
        # "The conditions are:" on its own is useless - glue the list onto it.
        if merged and merged[-1].endswith(":") and len(merged[-1]) < 300:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [s for s in merged if len(s) >= _MIN_SENTENCE_CHARS and not _is_heading(s)]


def _scored_sentences(question: str, sources: Sequence[Source]) -> list[tuple[float, str, Source]]:
    q_terms = set(_terms(question))
    scored: list[tuple[float, str, Source]] = []
    seen: set[str] = set()
    for src in sources:
        for sent in sentences(src.text):
            key = sent.lower()[:120]
            if key in seen:
                continue
            seen.add(key)
            words = set(_terms(sent))
            overlap = len(q_terms & words) / len(q_terms) if q_terms else 0.0
            if overlap <= 0:
                continue
            # Nudge by the passage's own retrieval score so the best page wins ties.
            scored.append((overlap + 0.25 * src.rerank_score, sent, src))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


_LIST_MARKER = re.compile(r"^\d+[.)]\s+")


def _clean(sent: str) -> str:
    """Drop a leading list marker so a bullet does not start with a number."""
    return _LIST_MARKER.sub("", sent)


def _bullet(sent: str, src: Source) -> str:
    return f"- {_clean(sent)} {src.citation}"


def compose(
    question: str,
    sources: Sequence[Source],
    *,
    exam_mode: bool = False,
    marks: int = 10,
    min_relevance: float = 0.20,
) -> str:
    if not sources:
        return NOT_FOUND_MESSAGE
    if max(s.rerank_score for s in sources) < min_relevance:
        return NOT_FOUND_MESSAGE

    scored = _scored_sentences(question, sources)
    if not scored:
        return NOT_FOUND_MESSAGE

    if not exam_mode:
        picks = scored[:4]
        return "\n".join(_bullet(sent, src) for _, sent, src in picks)

    n_points = 4 if marks <= 5 else 8
    used: set[str] = set()

    def take(pool, predicate=None, limit=1):
        chosen = []
        for _score, sent, src in pool:
            if sent in used:
                continue
            if predicate and not predicate(sent):
                continue
            used.add(sent)
            chosen.append((sent, src))
            if len(chosen) >= limit:
                break
        return chosen

    definition = take(scored, lambda s: any(c in s.lower() for c in _DEFINITION_CUES)) or take(
        scored
    )
    points = take(scored, limit=n_points)
    diagram = take(scored, lambda s: any(c in s.lower() for c in _DIAGRAM_CUES))
    conclusion = take(scored) or definition

    top = sources[0]
    lines = ["Definition"]
    lines += [f"{_clean(sent)} {src.citation}" for sent, src in definition] or [NOT_FOUND_MESSAGE]
    lines += ["", "Key points"]
    lines += [_bullet(sent, src) for sent, src in points] or ["- (no further points in the notes)"]
    lines += ["", "Diagram suggestion"]
    if diagram:
        sent, src = diagram[0]
        lines.append(
            f"Draw the structure described here and label it: {_clean(sent)} {src.citation}"
        )
    else:
        lines.append(
            f"Sketch a labelled diagram of the process described on p.{top.page} "
            f"of {top.file} {top.citation}"
        )
    lines += ["", "Conclusion"]
    lines += [f"{_clean(sent)} {src.citation}" for sent, src in conclusion]
    return "\n".join(lines)
