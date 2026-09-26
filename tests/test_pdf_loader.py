from __future__ import annotations

from pathlib import Path

from src.pdf_loader import (
    discover_pdfs,
    extract_pages,
    file_sha256,
    normalize_page_text,
    relative_source,
    subject_from_path,
)
from tests.conftest import MINI_PAGES


def test_extract_pages_keeps_page_numbers(mini_corpus):
    pdf = mini_corpus.pdf_dir / "DBMS" / "DBMS_Mini.pdf"
    pages = extract_pages(pdf)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "ACID properties" in pages[0].text
    assert "two phase locking" in pages[1].text.lower()
    assert "Write ahead logging" in pages[2].text


def test_subject_comes_from_the_folder_name(tmp_path: Path):
    root = tmp_path / "pdfs"
    assert subject_from_path(root / "DBMS" / "m2.pdf", root) == "DBMS"
    assert subject_from_path(root / "OS" / "x" / "m3.pdf", root) == "OS"
    assert subject_from_path(root / "loose.pdf", root) == "General"


def test_relative_source_is_posix(tmp_path: Path):
    root = tmp_path / "pdfs"
    assert relative_source(root / "DBMS" / "m2.pdf", root) == "DBMS/m2.pdf"


def test_hash_is_stable_and_content_sensitive(mini_corpus, make_pdf):
    pdf = mini_corpus.pdf_dir / "DBMS" / "DBMS_Mini.pdf"
    assert file_sha256(pdf) == file_sha256(pdf)

    other = make_pdf(mini_corpus.pdf_dir / "DBMS" / "Other.pdf", MINI_PAGES[:1])
    assert file_sha256(other) != file_sha256(pdf)


def test_normalize_rejoins_hyphenated_line_breaks():
    assert "normalization" in normalize_page_text("normal-\nization of tables")
    assert normalize_page_text("a\n\n\n\nb") == "a\n\nb"
    assert normalize_page_text("  lots   of    space ") == "lots of space"


def test_discover_pdfs_is_recursive_and_sorted(mini_corpus, make_pdf):
    make_pdf(mini_corpus.pdf_dir / "OS" / "OS_Mini.pdf", MINI_PAGES[:1])
    found = [p.name for p in discover_pdfs(mini_corpus.pdf_dir)]
    assert found == ["DBMS_Mini.pdf", "OS_Mini.pdf"]


def test_discover_pdfs_on_missing_folder(tmp_path: Path):
    assert discover_pdfs(tmp_path / "nope") == []
