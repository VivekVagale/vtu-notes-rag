"""Extract one PDF, in its own process, so it can be killed.

    python -m src.extract_worker <blob.pdf> <out.json> [--max-pages N] [--max-text-mb N]

Why a subprocess and not a thread: PyMuPDF does its work in C with the GIL
released, so a thread timeout returns while the thread stays wedged on a
malformed file forever. A subprocess can be killed. Embedding and the Chroma
write deliberately stay in the parent, because the index wants one writer.

Imports stay light (pymupdf + stdlib) - this process is spawned per document
and Windows has no fork.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.injection import scan  # noqa: E402
from src.pdf_loader import normalize_page_text  # noqa: E402
from src.sanitize import sanitize_with_report  # noqa: E402

MAX_PAGE_CHARS = 200_000


def extract(blob: Path, *, max_pages: int, max_text_mb: int) -> dict:
    try:
        import pymupdf
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # type: ignore[no-redef]

    try:
        doc = pymupdf.open(blob)
    except Exception as exc:
        return {"ok": False, "error_code": "unreadable_pdf", "error_detail": str(exc)[:300]}

    with doc:
        if doc.needs_pass or doc.is_encrypted:
            return {"ok": False, "error_code": "encrypted_pdf",
                    "error_detail": "an unreviewable document cannot be approved"}
        page_count = doc.page_count
        if page_count > max_pages:
            return {"ok": False, "error_code": "too_many_pages",
                    "error_detail": f"{page_count} pages, limit {max_pages}"}

        pages = []
        total_chars = 0
        pages_with_text = 0
        score = 0
        hard = False
        rules: set[str] = set()
        sanitized_counts = {"invisible": 0, "bidi": 0, "tags": 0, "fences": 0, "role_lines": 0}

        for number, page in enumerate(doc, start=1):
            raw = normalize_page_text(page.get_text("text"))
            text, report = sanitize_with_report(raw)
            if len(text) > MAX_PAGE_CHARS:
                return {"ok": False, "error_code": "rejected_oversize_text",
                        "error_detail": f"page {number} holds {len(text)} characters"}
            for key, value in report.as_dict().items():
                sanitized_counts[key] += value

            result = scan(text)
            score += result.score
            hard = hard or result.hard
            rules.update(result.rules)

            if text.strip():
                pages_with_text += 1
            total_chars += len(text)
            if total_chars > max_text_mb * 1024 * 1024:
                return {"ok": False, "error_code": "rejected_oversize_text",
                        "error_detail": f"more than {max_text_mb}MB of text"}

            pages.append({
                "page": number,
                "text": text,
                "injection": {"score": result.score, "hard": result.hard, "rules": result.rules},
            })

    body = "\n".join(p["text"] for p in pages)
    return {
        "ok": True,
        "page_count": page_count,
        "pages_with_text": pages_with_text,
        "text_chars": total_chars,
        "text_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "sanitize": sanitized_counts,
        "injection": {"score": score, "hard": hard, "rules": sorted(rules)},
        "pages": pages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="extract one PDF to JSON")
    parser.add_argument("blob")
    parser.add_argument("out")
    parser.add_argument("--max-pages", type=int, default=400)
    parser.add_argument("--max-text-mb", type=int, default=8)
    args = parser.parse_args(argv)

    try:
        payload = extract(Path(args.blob), max_pages=args.max_pages, max_text_mb=args.max_text_mb)
    except Exception as exc:  # never let the parent see a bare traceback
        payload = {"ok": False, "error_code": "extract_crashed", "error_detail": str(exc)[:300]}

    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
