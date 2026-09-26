"""Retrieval evaluation: does the right page come back in the top-k?

    python -m src.evaluate            # uses eval/questions.json
    python -m src.evaluate --k 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_settings
from .retrieve import Retriever

DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "questions.json"


@dataclass
class QuestionSpec:
    id: str
    question: str
    expected_file: str
    expected_pages: list[int]
    subject: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "QuestionSpec":
        pages = raw.get("expected_pages")
        if pages is None and "expected_page" in raw:
            pages = [raw["expected_page"]]
        return cls(
            id=str(raw.get("id", raw["question"][:24])),
            question=raw["question"],
            expected_file=raw["expected_file"],
            expected_pages=[int(p) for p in (pages or [])],
            subject=raw.get("subject"),
        )


@dataclass
class QuestionResult:
    spec: QuestionSpec
    hit: bool
    rank: int | None
    retrieved: list[str] = field(default_factory=list)


def load_questions(path: Path | str | None = None) -> list[QuestionSpec]:
    path = Path(path) if path else DEFAULT_QUESTIONS
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    return [QuestionSpec.from_dict(item) for item in items]


def evaluate(
    retriever: Retriever,
    questions: list[QuestionSpec],
    k: int = 5,
    use_subject_filter: bool = True,
) -> dict[str, Any]:
    results: list[QuestionResult] = []
    for spec in questions:
        sources = retriever.retrieve(
            spec.question, k=k, subject=spec.subject if use_subject_filter else None
        )
        rank: int | None = None
        for position, src in enumerate(sources, start=1):
            if (
                src.file.lower() == spec.expected_file.lower()
                and src.page in spec.expected_pages
            ):
                rank = position
                break
        results.append(
            QuestionResult(
                spec=spec,
                hit=rank is not None,
                rank=rank,
                retrieved=[f"{s.file} p.{s.page}" for s in sources],
            )
        )

    hits = [r for r in results if r.hit]
    mrr = sum(1.0 / r.rank for r in hits if r.rank) / len(results) if results else 0.0
    return {
        "k": k,
        "total": len(results),
        "hits": len(hits),
        "hit_rate": len(hits) / len(results) if results else 0.0,
        "mrr": mrr,
        "results": results,
    }


def print_report(report: dict[str, Any]) -> None:
    k = report["k"]
    print(f"{'id':<22} {'hit':<5} {'rank':<5} top-1 retrieved")
    print("-" * 78)
    for result in report["results"]:
        top1 = result.retrieved[0] if result.retrieved else "(nothing)"
        print(
            f"{result.spec.id:<22} {'YES' if result.hit else 'no':<5} "
            f"{str(result.rank or '-'):<5} {top1}"
        )
    print("-" * 78)
    print(
        f"hit rate @{k}: {report['hits']}/{report['total']} "
        f"= {report['hit_rate'] * 100:.1f}%   MRR: {report['mrr']:.3f}"
    )
    misses = [r for r in report["results"] if not r.hit]
    if misses:
        print("\nMisses:")
        for result in misses:
            expected = f"{result.spec.expected_file} p.{result.spec.expected_pages}"
            print(f"  {result.spec.id}: expected {expected}")
            for entry in result.retrieved[:3]:
                print(f"      got {entry}")


def run_eval(
    retriever: Retriever | None = None,
    *,
    questions_path: Path | str | None = None,
    k: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    settings = load_settings()
    retriever = retriever or Retriever(settings)
    questions = load_questions(questions_path)
    report = evaluate(retriever, questions, k=k or settings.top_k)
    if verbose:
        print_report(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieval hit-rate evaluation")
    parser.add_argument("--file", default=None, help="questions JSON (default eval/questions.json)")
    parser.add_argument("--k", type=int, default=None, help="top-k to score against")
    args = parser.parse_args(argv)
    report = run_eval(questions_path=args.file, k=args.k, verbose=True)
    return 0 if report["hit_rate"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
