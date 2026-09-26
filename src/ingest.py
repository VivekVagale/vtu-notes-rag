"""Ingestion pipeline: PDF -> pages -> chunks -> embeddings -> Chroma.

Files whose SHA-256 is already in the manifest are skipped; changed files have
their old chunks dropped before re-indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunking import BaseTokenizer, chunk_text, get_tokenizer
from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .manifest import Manifest
from .pdf_loader import (
    discover_pdfs,
    extract_pages,
    file_sha256,
    relative_source,
    subject_from_path,
)
from .store import ChunkRecord, VectorStore

Reporter = Callable[[str], None]


@dataclass
class IngestStats:
    files_seen: int = 0
    files_indexed: int = 0
    files_updated: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    files_empty: int = 0
    pages: int = 0
    chunks: int = 0

    def summary(self) -> str:
        return (
            f"{self.files_seen} PDF(s) seen | {self.files_indexed} new, "
            f"{self.files_updated} updated, {self.files_skipped} unchanged, "
            f"{self.files_removed} removed, {self.files_empty} without text | "
            f"{self.pages} pages, {self.chunks} chunks"
        )


def chunk_id(source: str, page: int, index: int) -> str:
    return f"{source}#p{page}#c{index}"


def build_records(
    pdf_path: Path,
    *,
    settings: Settings,
    tokenizer: BaseTokenizer,
) -> tuple[list[ChunkRecord], int, str, str]:
    """Return (records, page_count, subject, file_hash) for one PDF."""
    source = relative_source(pdf_path, settings.pdf_dir)
    subject = subject_from_path(pdf_path, settings.pdf_dir)
    file_hash = file_sha256(pdf_path)
    pages = extract_pages(pdf_path)

    records: list[ChunkRecord] = []
    for page in pages:
        if not page.text.strip():
            continue
        chunks = chunk_text(
            page.text,
            max_tokens=settings.chunk_tokens,
            overlap=settings.chunk_overlap,
            tokenizer=tokenizer,
            min_chars=settings.min_chunk_chars,
        )
        for chunk in chunks:
            records.append(
                ChunkRecord(
                    id=chunk_id(source, page.page_number, chunk.index),
                    text=chunk.text,
                    metadata={
                        "file": pdf_path.name,
                        "source": source,
                        "subject": subject,
                        "page": page.page_number,
                        "chunk_index": chunk.index,
                        "file_hash": file_hash,
                        "chars": len(chunk.text),
                    },
                )
            )
    return records, len(pages), subject, file_hash


def ingest(
    settings: Settings | None = None,
    *,
    force: bool = False,
    subject: str | None = None,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    report: Reporter | None = None,
) -> IngestStats:
    """Index every PDF under settings.pdf_dir that is new or has changed."""
    settings = settings or get_settings()
    store = store or VectorStore(settings)
    embedder = embedder or get_embedder(settings)
    say: Reporter = report or (lambda _msg: None)

    tokenizer = get_tokenizer(settings.embed_model if settings.embed_backend != "hash" else None)
    manifest = Manifest.load(settings.manifest_path)
    stats = IngestStats()

    all_pdfs = discover_pdfs(settings.pdf_dir)
    if not all_pdfs:
        say(f"[warn] no PDFs found under {settings.pdf_dir}")

    # Forget files that were deleted from disk.
    known = {relative_source(p, settings.pdf_dir) for p in all_pdfs}
    for source in manifest.prune(known):
        store.delete_by_source(source)
        stats.files_removed += 1
        say(f"[gone] {source} - removed from index")

    pdfs = all_pdfs
    if subject:
        pdfs = [p for p in pdfs if subject_from_path(p, settings.pdf_dir).lower() == subject.lower()]

    for pdf_path in pdfs:
        stats.files_seen += 1
        source = relative_source(pdf_path, settings.pdf_dir)
        digest = file_sha256(pdf_path)
        state = manifest.status(source, digest)

        if state == "unchanged" and not force:
            stats.files_skipped += 1
            say(f"[skip] {source} (unchanged)")
            continue

        if state != "new" or force:
            store.delete_by_source(source)  # no duplicate chunks on re-index

        records, page_count, subj, digest = build_records(
            pdf_path, settings=settings, tokenizer=tokenizer
        )
        if not records:
            stats.files_empty += 1
            say(f"[warn] {source} has no extractable text (scanned PDF?) - skipped")
            continue

        vectors = embedder.embed_documents([r.text for r in records])
        store.add(records, vectors)
        manifest.record(
            source, file_hash=digest, subject=subj, pages=page_count, chunks=len(records)
        )
        stats.pages += page_count
        stats.chunks += len(records)
        if state == "new":
            stats.files_indexed += 1
        else:
            stats.files_updated += 1
        say(f"[ok]   {source} -> {len(records)} chunks from {page_count} pages [{subj}]")

    manifest.save()
    return stats
