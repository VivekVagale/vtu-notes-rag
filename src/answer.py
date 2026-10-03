"""Glue: question -> retrieval -> grounded, cited answer."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .config import NOT_FOUND_MESSAGE, Settings, get_settings
from .extractive import compose, usable_sources
from .ingest import IngestStats, ingest
from .llm import LLMError, LLMProvider, get_provider
from .prompts import system_prompt, user_prompt
from .store import PUBLIC_TIERS
from .retrieve import Retriever, Source

CITATION_RE = re.compile(r"\[([^\[\]]+?\.pdf),\s*p\.\s*(\d+)\]", re.IGNORECASE)

# Contact details that must not ride out of a stranger's PDF into an answer.
URL_RE = re.compile(r"https?://\S+|\bwww\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+\d{1,3}[ -]?)?\d{10}(?!\d)")
REDACTED = "[removed]"



@dataclass
class Answer:
    question: str
    text: str
    sources: list[Source]
    provider: str
    model: str
    exam_mode: bool = False
    marks: int = 10
    subject: str | None = None
    scopes: list[str] | None = None
    elapsed_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def not_found(self) -> bool:
        return self.text.strip().lower().startswith(NOT_FOUND_MESSAGE.lower())

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "not_found": self.not_found,
            "provider": self.provider,
            "model": self.model,
            "exam_mode": self.exam_mode,
            "marks": self.marks,
            "subject": self.subject,
            "scopes": self.scopes,
            "elapsed_s": round(self.elapsed_s, 2),
            "warnings": self.warnings,
            "sources": [s.to_dict() for s in self.sources],
        }


def contacts_in(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in (URL_RE, EMAIL_RE, PHONE_RE):
        found.update(m.group(0) for m in pattern.finditer(text))
    return found


def scrub_contributed_contacts(
    text: str, sources: Sequence[Source]
) -> tuple[str, list[str]]:
    """Strip URLs, emails and phone numbers that came from contributed notes.

    A link inside somebody else's upload must not reach a reader wearing this
    site's voice. The owner's own curated notes are left alone.
    """
    risky: set[str] = set()
    for source in sources:
        if source.visibility == "community":
            risky.update(contacts_in(source.text))
    if not risky:
        return text, []

    removed = []
    for item in sorted(risky, key=len, reverse=True):
        if item in text:
            text = text.replace(item, REDACTED)
            removed.append(item)
    warnings = (
        [f"removed {len(removed)} contact detail(s) carried in contributed notes"]
        if removed
        else []
    )
    return text, warnings


def validate_citations(text: str, sources: Sequence[Source]) -> list[str]:
    """Flag citations that point at pages the retriever never returned."""
    allowed = {(s.file.lower(), s.page) for s in sources}
    warnings: list[str] = []
    for file_name, page in CITATION_RE.findall(text):
        key = (file_name.strip().lower(), int(page))
        if key not in allowed:
            warnings.append(f"cited [{file_name}, p.{page}] is not in the retrieved passages")
    if text.strip() and not CITATION_RE.search(text) and not text.strip().lower().startswith(
        NOT_FOUND_MESSAGE.lower()
    ):
        warnings.append("answer contains no inline citation")
    return sorted(set(warnings))


class RagEngine:
    """Everything a UI needs, with lazily built heavy parts."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: Retriever | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(self.settings)
        self._provider = provider
        self._provider_ready = provider is not None

    @property
    def provider(self) -> LLMProvider | None:
        if not self._provider_ready:
            self._provider = get_provider(self.settings)
            self._provider_ready = True
        return self._provider

    @property
    def provider_name(self) -> str:
        return self.settings.llm_provider

    def ask(
        self,
        question: str,
        *,
        subject: str | None = None,
        k: int | None = None,
        exam_mode: bool = False,
        marks: int = 10,
        scopes: Sequence[str] | None = None,
        owner_id: str | None = None,
    ) -> Answer:
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise ValueError("question is empty")

        # Record the tiers actually searched, not the ones asked for.
        effective = list(scopes) if scopes is not None else list(PUBLIC_TIERS)
        sources = self.retriever.retrieve(
            question, k=k, subject=subject, scopes=scopes, owner_id=owner_id
        )
        model = "-"

        if not sources:
            return Answer(
                question=question,
                text=NOT_FOUND_MESSAGE,
                sources=[],
                provider=self.provider_name,
                model=model,
                exam_mode=exam_mode,
                marks=marks,
                subject=subject,
                scopes=effective,
                elapsed_s=time.perf_counter() - started,
            )

        provider = self.provider
        extra_warnings: list[str] = []
        if provider is None:  # extractive
            allow = self.settings.extractive_community
            quotable = usable_sources(sources, allow_community=allow)
            dropped = len(sources) - len(quotable)
            if dropped:
                extra_warnings.append(
                    f"{dropped} contributed passage(s) left out: the extractive "
                    "provider quotes verbatim. Set EXTRACTIVE_COMMUNITY=true or "
                    "use a real LLM provider to include them."
                )
                sources = quotable
            if not sources:
                return Answer(
                    question=question,
                    text=NOT_FOUND_MESSAGE,
                    sources=[],
                    provider=self.provider_name,
                    model="extractive",
                    exam_mode=exam_mode,
                    marks=marks,
                    subject=subject,
                    scopes=effective,
                    elapsed_s=time.perf_counter() - started,
                    warnings=extra_warnings,
                )
            text = compose(
                question, sources, exam_mode=exam_mode, marks=marks, allow_community=allow
            )
            model = "extractive"
        else:
            model = provider.model
            text = provider.generate(
                system_prompt(exam_mode=exam_mode, marks=marks),
                user_prompt(question, sources, exam_mode=exam_mode, marks=marks),
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )
            if not text.strip():
                raise LLMError("provider returned an empty response")

        text, scrub_warnings = scrub_contributed_contacts(text, sources)
        return Answer(
            question=question,
            text=text.strip(),
            sources=sources,
            provider=self.provider_name,
            model=model,
            exam_mode=exam_mode,
            marks=marks,
            subject=subject,
            scopes=effective,
            elapsed_s=time.perf_counter() - started,
            warnings=extra_warnings + scrub_warnings + validate_citations(text, sources),
        )

    # --- passthroughs ---------------------------------------------------
    def ingest(self, **kwargs: Any) -> IngestStats:
        return ingest(
            self.settings,
            store=self.retriever.store,
            embedder=self.retriever.embedder,
            **kwargs,
        )

    def subjects(self) -> list[str]:
        return self.retriever.subjects()

    def stats(self) -> dict[str, Any]:
        info = self.settings.describe()
        info["chunks_indexed"] = self.retriever.count()
        info["subjects"] = ", ".join(self.subjects()) or "-"
        return info
