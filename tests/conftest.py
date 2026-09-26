"""Shared fixtures. Tests run fully offline: hash embeddings, extractive answers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # type: ignore[no-redef]

from src.config import Settings, load_settings  # noqa: E402

MINI_PAGES = [
    (
        "Unit 1 - Transactions",
        "A transaction is a logical unit of work that accesses and possibly modifies "
        "the contents of a database. The ACID properties are atomicity, consistency, "
        "isolation and durability. Atomicity means all operations of the transaction "
        "complete or none of them do.",
    ),
    (
        "Unit 2 - Concurrency Control",
        "Two phase locking protocol has a growing phase in which locks are acquired "
        "and a shrinking phase in which locks are released. Strict two phase locking "
        "holds every exclusive lock until the transaction commits, which avoids "
        "cascading rollback.",
    ),
    (
        "Unit 3 - Recovery",
        "Write ahead logging requires that the log record for an update reaches stable "
        "storage before the data page does. Checkpointing limits the amount of log that "
        "must be scanned during recovery after a crash.",
    ),
]


def write_pdf(path: Path, pages: list[tuple[str, str]]) -> Path:
    """Minimal text PDF writer used to build fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for heading, body in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 110), heading, fontsize=14, fontname="hebo")
        page.insert_textbox(pymupdf.Rect(50, 110, 545, 780), body, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def make_pdf():
    return write_pdf


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    return load_settings(
        pdf_dir=tmp_path / "pdfs",
        index_dir=tmp_path / "index",
        collection="test_notes",
        embed_backend="hash",
        llm_provider="extractive",
        chunk_tokens=120,
        chunk_overlap=20,
        min_chunk_chars=20,
        top_k=3,
        fetch_k=12,
        reranker="lexical",
    )


@pytest.fixture
def mini_corpus(tmp_settings: Settings) -> Settings:
    """One three-page DBMS PDF inside the temporary pdf_dir."""
    write_pdf(tmp_settings.pdf_dir / "DBMS" / "DBMS_Mini.pdf", MINI_PAGES)
    return tmp_settings
