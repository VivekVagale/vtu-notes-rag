"""Neutralise text tricks in PDFs before the text is ever chunked or shown.

Deliberately a separate module from pdf_loader.normalize_page_text: that one
tidies extractor noise for *every* PDF including your own, this one defends
against text that was written to manipulate a reader or a model. Keeping them
apart means the curated path is unchanged and testable as it was.

Pure stdlib, no heavy imports - this runs inside the extraction subprocess.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Invisible characters: zero-width joiners/spaces, word joiner, BOM.
_INVISIBLE = re.compile("[​-‏⁠-⁤﻿­]")
# Bidi overrides - can make rendered text read differently from its bytes.
_BIDI = re.compile("[‪-‮⁦-⁩]")
# Unicode tag block: a whole alphabet that renders as nothing at all.
_TAGS = re.compile("[\U000e0000-\U000e007f]")
# Long rules of = or - impersonate a section fence in the prompt.
_FENCE = re.compile(r"(?m)^([=\-_*#~`]){4,}\s*$")
# A line that opens like a chat turn.
_ROLE_LINE = re.compile(
    r"(?im)^[ \t>*-]*(system|assistant|user|human|ai|chatgpt|claude)\s*:",
)


@dataclass
class SanitizeReport:
    invisible: int = 0
    bidi: int = 0
    tags: int = 0
    fences: int = 0
    role_lines: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.invisible or self.bidi or self.tags or self.fences or self.role_lines)

    def as_dict(self) -> dict[str, int]:
        return {
            "invisible": self.invisible,
            "bidi": self.bidi,
            "tags": self.tags,
            "fences": self.fences,
            "role_lines": self.role_lines,
        }


def sanitize_text(text: str) -> str:
    """Return text with invisible, bidi and turn-shaped tricks defused."""
    return sanitize_with_report(text)[0]


def sanitize_with_report(text: str) -> tuple[str, SanitizeReport]:
    report = SanitizeReport()
    if not text:
        return "", report

    report.invisible = len(_INVISIBLE.findall(text))
    text = _INVISIBLE.sub("", text)

    report.bidi = len(_BIDI.findall(text))
    text = _BIDI.sub("", text)

    report.tags = len(_TAGS.findall(text))
    text = _TAGS.sub("", text)

    # NFKC folds look-alike codepoints so a rule cannot be dodged with fullwidth
    # or mathematical letters. Done after stripping, so nothing folds into a
    # character we meant to remove.
    text = unicodedata.normalize("NFKC", text)

    report.fences = len(_FENCE.findall(text))
    text = _FENCE.sub("---", text)

    # "System:" at the start of a line reads as a turn boundary. Rewrite rather
    # than delete: real notes do say "system:" and the sentence must survive.
    def _derole(match: re.Match[str]) -> str:
        report.role_lines += 1
        return f"{match.group(1)} -"

    text = _ROLE_LINE.sub(_derole, text)
    return text, report


def safe_filename(name: str, *, fallback: str = "document.pdf", limit: int = 80) -> str:
    """A display name that cannot carry a path, a quote or a control character."""
    cleaned = re.sub(r"[^A-Za-z0-9 ._-]", "", name or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lstrip(".")
    if len(cleaned) > limit:
        stem, _, _ext = cleaned.rpartition(".")
        cleaned = (stem or cleaned)[: limit - 4].strip() + ".pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf" if cleaned else fallback
    return cleaned or fallback
