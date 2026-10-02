"""Detect text in an uploaded PDF that is aimed at the answering model.

Runs ONCE at ingest, never at query time. Pure stdlib regex - it executes inside
the extraction subprocess, so it must not import anything heavy.

Design rules, learned the hard way:
- HARD families must be things that cannot occur in real engineering notes. A
  VTU Operating Systems module contains the line "system:" and a DBMS module
  legitimately says "ignore the previous constraint" - neither may be a hard hit.
  There is a regression test asserting zero hard hits across the whole curated
  corpus, and it is the gate on ever letting this block anything.
- Soft families only raise a score for a human to look at. They never block.
- Quarantine is expressed as a tier the scope allowlist does not include, never
  as a numeric exclusion filter: on Chroma, {"score": {"$lt": 3}} drops every row
  that lacks the key, which would take the entire existing corpus dark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HARD_SCORE = 5
SOFT_SCORE = 1
WINDOW = 100


@dataclass(frozen=True)
class Rule:
    rule_id: str
    family: str
    hard: bool
    pattern: re.Pattern[str]
    why: str


def _c(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


RULES: tuple[Rule, ...] = (
    # --- A: instruction override. Requires an explicit instruction noun, so
    # "ignore the previous constraint" in a normalization example is not a hit.
    Rule(
        "A1",
        "override",
        True,
        _c(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|preceding|all|any)\b[^.\n]{0,40}"
            r"\b(instruction|instructions|prompt|prompts|rule|rules|direction|directions|"
            r"guideline|guidelines|system\s+message)\b"
        ),
        "tells the model to drop its instructions",
    ),
    Rule(
        "A2",
        "override",
        True,
        _c(r"\b(your|the)\s+(real|true|actual|new)\s+(instruction|instructions|task|purpose)\b"),
        "claims to replace the real instructions",
    ),
    # --- B: turn/template forgery. Chat control tokens never occur in notes.
    Rule(
        "B1",
        "turn-forgery",
        True,
        _c(r"<\|\s*(im_start|im_end|system|user|assistant|endoftext)\s*\|>"),
        "chat template control token",
    ),
    Rule("B2", "turn-forgery", True, _c(r"\[/?INST\]|<<SYS>>|<</SYS>>"), "llama turn marker"),
    Rule(
        "B3",
        "turn-forgery",
        True,
        _c(r"^#{2,}\s*(instruction|system|assistant)\b"),
        "markdown heading posing as a turn header",
    ),
    Rule(
        "B4",
        "turn-forgery",
        True,
        _c(r"^\s*(human|assistant)\s*:\s*$"),
        "bare turn label on its own line",
    ),
    # --- C: persona hijack (soft - teaching material does say "act as").
    Rule(
        "C1",
        "persona",
        False,
        _c(r"\byou\s+are\s+(now|no\s+longer)\b|\bfrom\s+now\s+on\b[^.\n]{0,30}\byou\b"),
        "attempts to reassign the model's role",
    ),
    Rule("C2", "persona", False, _c(r"\b(developer|debug|god)\s+mode\b"), "mode-switch claim"),
    # --- D: exfiltration.
    Rule(
        "D1",
        "exfiltration",
        False,
        _c(r"\b(send|post|upload|email|forward|exfiltrate)\b[^.\n]{0,30}\b(to|at)\b[^.\n]{0,20}https?://"),
        "asks for data to be sent somewhere",
    ),
    Rule(
        "D2",
        "exfiltration",
        False,
        _c(r"\b(curl|wget|fetch)\s+https?://"),
        "asks the reader to fetch a URL",
    ),
    # --- E: output hijack.
    Rule(
        "E1",
        "output-hijack",
        False,
        _c(r"\b(respond|reply|answer|output)\b[^.\n]{0,20}\bonly\s+with\b"),
        "dictates the answer format",
    ),
    Rule(
        "E2",
        "output-hijack",
        False,
        _c(r"\b(do\s+not|don't|never)\b[^.\n]{0,20}\b(cite|mention|reveal|tell)\b"),
        "asks the model to hide something",
    ),
    # --- F: payloads.
    Rule("F1", "payload", False, _c(r"<\s*script\b|javascript\s*:"), "script payload"),
    Rule(
        "F2",
        "payload",
        False,
        _c(r"\brm\s+-rf\s+/|\bDROP\s+TABLE\b|\bDELETE\s+FROM\b[^.\n]{0,20};"),
        "destructive command",
    ),
)


@dataclass
class Hit:
    rule_id: str
    family: str
    hard: bool
    why: str
    start: int
    window: str


@dataclass
class Scan:
    score: int = 0
    hard: bool = False
    rules: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.hard or self.score >= 3

    def as_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "hard": self.hard,
            "rules": self.rules,
            "hits": [
                {"rule": h.rule_id, "family": h.family, "at": h.start, "window": h.window}
                for h in self.hits
            ],
        }


def scan(text: str, *, max_hits_per_rule: int = 3) -> Scan:
    """Score text for prompt-injection signals. Never raises, never blocks."""
    result = Scan()
    if not text:
        return result
    for rule in RULES:
        found = 0
        for match in rule.pattern.finditer(text):
            if found >= max_hits_per_rule:
                break
            found += 1
            start = max(0, match.start() - WINDOW // 2)
            result.hits.append(
                Hit(
                    rule_id=rule.rule_id,
                    family=rule.family,
                    hard=rule.hard,
                    why=rule.why,
                    start=match.start(),
                    window=" ".join(text[start : match.end() + WINDOW // 2].split()),
                )
            )
        if found:
            result.rules.append(rule.rule_id)
            result.score += (HARD_SCORE if rule.hard else SOFT_SCORE) * found
            result.hard = result.hard or rule.hard
    return result
