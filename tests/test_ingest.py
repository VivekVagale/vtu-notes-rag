from __future__ import annotations

from src.embeddings import HashEmbedder
from src.ingest import ingest
from src.manifest import Manifest
from src.store import VectorStore
from tests.conftest import MINI_PAGES


def _run(settings, **kwargs):
    store = VectorStore(settings)
    stats = ingest(settings, store=store, embedder=HashEmbedder(), **kwargs)
    return stats, store


def test_first_ingest_indexes_every_page(mini_corpus):
    stats, store = _run(mini_corpus)
    assert stats.files_indexed == 1
    assert stats.pages == 3
    assert stats.chunks >= 3
    assert store.count() == stats.chunks

    metas = store._all_metadatas()
    assert {m["subject"] for m in metas} == {"DBMS"}
    assert {m["file"] for m in metas} == {"DBMS_Mini.pdf"}
    assert sorted({m["page"] for m in metas}) == [1, 2, 3]
    assert all(m["source"] == "DBMS/DBMS_Mini.pdf" for m in metas)


def test_second_ingest_skips_unchanged_files(mini_corpus):
    stats_one, store = _run(mini_corpus)
    stats_two, store_two = _run(mini_corpus)
    assert stats_two.files_skipped == 1
    assert stats_two.chunks == 0
    assert store_two.count() == stats_one.chunks  # no duplicates


def test_force_reindexes_without_duplicating(mini_corpus):
    stats_one, _ = _run(mini_corpus)
    stats_two, store = _run(mini_corpus, force=True)
    assert stats_two.files_updated == 1
    assert store.count() == stats_one.chunks


def test_changed_file_is_reindexed(mini_corpus, make_pdf):
    _run(mini_corpus)
    pdf = mini_corpus.pdf_dir / "DBMS" / "DBMS_Mini.pdf"
    make_pdf(pdf, MINI_PAGES + [("Unit 4 - Indexing", "A B+ tree index keeps all keys in leaves.")])

    stats, store = _run(mini_corpus)
    assert stats.files_updated == 1
    assert 4 in {m["page"] for m in store._all_metadatas()}


def test_deleted_pdf_is_removed_from_the_index(mini_corpus, make_pdf):
    make_pdf(mini_corpus.pdf_dir / "OS" / "OS_Mini.pdf", MINI_PAGES[:1])
    _run(mini_corpus)

    (mini_corpus.pdf_dir / "OS" / "OS_Mini.pdf").unlink()
    stats, store = _run(mini_corpus)

    assert stats.files_removed == 1
    assert store.subjects() == ["DBMS"]
    assert "OS/OS_Mini.pdf" not in Manifest.load(mini_corpus.manifest_path).entries


def test_subject_filter_limits_ingestion(mini_corpus, make_pdf):
    make_pdf(mini_corpus.pdf_dir / "OS" / "OS_Mini.pdf", MINI_PAGES[:1])
    stats, store = _run(mini_corpus, subject="OS")
    assert stats.files_indexed == 1
    assert store.subjects() == ["OS"]


def test_manifest_records_hash_and_counts(mini_corpus):
    _run(mini_corpus)
    manifest = Manifest.load(mini_corpus.manifest_path)
    entry = manifest.entries["DBMS/DBMS_Mini.pdf"]
    assert len(entry["hash"]) == 64
    assert entry["subject"] == "DBMS"
    assert entry["pages"] == 3
    assert entry["chunks"] >= 3
