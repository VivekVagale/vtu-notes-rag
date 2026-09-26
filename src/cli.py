"""Command line interface.

    python -m src.cli ingest [--force] [--subject DBMS]
    python -m src.cli ask "What is 3NF?" [--subject DBMS] [-k 5] [--exam] [--marks 10]
    python -m src.cli subjects
    python -m src.cli status
    python -m src.cli eval [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys

from .answer import Answer, RagEngine
from .config import load_settings
from .llm import LLMError

_RULE = "-" * 72


def _engine(args: argparse.Namespace) -> RagEngine:
    overrides = {}
    if getattr(args, "provider", None):
        overrides["llm_provider"] = args.provider.lower()
    if getattr(args, "subject_dir", None):
        overrides["pdf_dir"] = args.subject_dir
    return RagEngine(load_settings(**overrides))


def _print_answer(answer: Answer, show_sources: bool = True) -> None:
    print(_RULE)
    print(f"Q: {answer.question}")
    print(_RULE)
    print(answer.text)
    if answer.warnings:
        print()
        for warning in answer.warnings:
            print(f"[warn] {warning}")
    if show_sources and answer.sources:
        print()
        print("Sources")
        for i, src in enumerate(answer.sources, start=1):
            print(f"  {i}. {src.file}  p.{src.page}  [{src.subject}]  score={src.rerank_score:.3f}")
            print(f"     {src.snippet(220)}")
    print()
    print(
        f"[provider={answer.provider} model={answer.model} "
        f"sources={len(answer.sources)} {answer.elapsed_s:.2f}s]"
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    engine = _engine(args)
    print(f"Indexing PDFs under {engine.settings.pdf_dir}")
    stats = engine.ingest(force=args.force, subject=args.subject, report=print)
    print(_RULE)
    print(stats.summary())
    print(f"Collection now holds {engine.retriever.count()} chunks.")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    engine = _engine(args)
    if engine.retriever.count() == 0:
        print("Index is empty. Run: python -m src.cli ingest", file=sys.stderr)
        return 1
    try:
        answer = engine.ask(
            args.question,
            subject=args.subject,
            k=args.k,
            exam_mode=args.exam,
            marks=args.marks,
        )
    except LLMError as exc:
        print(f"[llm error] {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(answer.to_dict(), indent=2))
    else:
        _print_answer(answer, show_sources=not args.hide_sources)
    return 0


def cmd_subjects(args: argparse.Namespace) -> int:
    engine = _engine(args)
    subjects = engine.subjects()
    if not subjects:
        print("Nothing indexed yet. Run: python -m src.cli ingest")
        return 1
    for subject in subjects:
        files = [f for f in engine.retriever.store.indexed_files() if f["subject"] == subject]
        print(f"{subject} ({len(files)} file(s))")
        for entry in files:
            print(f"   - {entry['file']}  {entry['chunks']} chunks, {entry['pages']} pages")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    engine = _engine(args)
    for key, value in engine.stats().items():
        print(f"{key:>16}: {value}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import run_eval

    engine = _engine(args)
    report = run_eval(engine.retriever, questions_path=args.file, k=args.k, verbose=True)
    return 0 if report["hit_rate"] > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="RAG Q&A over your VTU syllabus/notes PDFs, with page citations.",
    )
    parser.add_argument(
        "--provider",
        help="override LLM_PROVIDER for this run (anthropic|ollama|extractive)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index new or changed PDFs")
    p_ingest.add_argument("--force", action="store_true", help="re-index even if unchanged")
    p_ingest.add_argument("--subject", help="only this subject folder")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--subject", help="restrict retrieval to one subject folder")
    p_ask.add_argument("-k", type=int, default=None, help="number of passages to use")
    p_ask.add_argument("--exam", action="store_true", help="VTU exam-style answer")
    p_ask.add_argument("--marks", type=int, default=10, choices=[5, 10], help="exam answer length")
    p_ask.add_argument("--json", action="store_true", help="machine readable output")
    p_ask.add_argument("--hide-sources", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_subjects = sub.add_parser("subjects", help="list indexed subjects and files")
    p_subjects.set_defaults(func=cmd_subjects)

    p_status = sub.add_parser("status", help="show configuration and index size")
    p_status.set_defaults(func=cmd_status)

    p_eval = sub.add_parser("eval", help="retrieval hit-rate evaluation")
    p_eval.add_argument("--file", default=None, help="path to questions JSON")
    p_eval.add_argument("--k", type=int, default=None)
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
