"""PDF -> per-page text, plus the file bookkeeping ingestion needs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

try:  # PyMuPDF >= 1.24 prefers the `pymupdf` name
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore[no-redef]

_SOFT_HYPHEN = "­"
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_HSPACE = re.compile(r"[ \t   ]+")
_BLANK_RUN = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based, matches what the PDF reader shows
    text: str


def file_sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_from_path(pdf_path: Path, pdf_root: Path) -> str:
    """data/pdfs/DBMS/Module2.pdf -> 'DBMS'. Loose files -> 'General'."""
    try:
        rel = pdf_path.resolve().relative_to(pdf_root.resolve())
    except ValueError:
        return pdf_path.parent.name or "General"
    return rel.parts[0] if len(rel.parts) > 1 else "General"


def relative_source(pdf_path: Path, pdf_root: Path) -> str:
    try:
        return pdf_path.resolve().relative_to(pdf_root.resolve()).as_posix()
    except ValueError:
        return pdf_path.name


def normalize_page_text(raw: str) -> str:
    """Tidy extractor noise without destroying layout cues."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace(_SOFT_HYPHEN, "")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words split across lines
    text = _HSPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[PageText]:
    """Extract text page by page. Pages with no text layer come back empty."""
    pages: list[PageText] = []
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, start=1):
            pages.append(PageText(number, normalize_page_text(page.get_text("text"))))
    return pages


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        return []
    return sorted(p for p in pdf_dir.rglob("*.pdf") if p.is_file() and not p.name.startswith("~"))
