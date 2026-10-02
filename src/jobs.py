"""The ingest worker: one job at a time, inside the API process.

Extraction is a killable subprocess; embedding and the Chroma write stay here so
the index keeps a single writer. A job is claimed atomically, so a second worker
(or a restart racing the old one) cannot take the same document twice.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .blobs import LocalBlobStore, get_blob_store
from .chunking import chunk_text, get_tokenizer
from .config import Settings
from .embeddings import Embedder, get_embedder
from .registry import (
    JOB_QUEUED,
    STATE_EXTRACTING,
    STATE_FAILED,
    STATE_INDEXED,
    STATE_UPLOADED,
    Document,
    Job,
    SqliteRegistry,
    StateConflict,
    get_registry,
    utcnow,
)
from .store import ChunkRecord, VectorStore

#: Below this share of pages carrying text, the file is a scan we cannot cite.
MIN_TEXT_PAGE_RATIO = 0.6
EXTRACT_TIMEOUT_S = 30
MAX_CHUNKS = 6000


@dataclass
class JobOutcome:
    job_id: str
    doc_id: str
    ok: bool
    chunks: int = 0
    error_code: str | None = None
    quarantined: bool = False


class IngestWorker:
    def __init__(
        self,
        settings: Settings,
        *,
        registry: SqliteRegistry | None = None,
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        blobs: LocalBlobStore | None = None,
    ) -> None:
        self.settings = settings
        # Its own registry connection: SQLite connections are not shared across
        # threads, and WAL lets this one write while the API reads.
        self.registry = registry or get_registry(settings)
        self.store = store or VectorStore(settings)
        self.embedder = embedder or get_embedder(settings)
        self.blobs = blobs or get_blob_store(settings)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- the loop -------------------------------------------------------
    def start(self) -> None:
        if self._thread:
            return
        self.registry.requeue_running_jobs(now=utcnow())
        self._thread = threading.Thread(target=self._loop, name="ingest-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:  # pragma: no cover - exercised through run_once
        while not self._stop.is_set():
            if self.run_once() is None:
                self._stop.wait(0.25)

    # --- one job --------------------------------------------------------
    def run_once(self) -> JobOutcome | None:
        job = self.registry.claim_job(now=utcnow())
        if job is None:
            return None
        document = self.registry.get_document(job.doc_id)
        if document is None:
            self.registry.fail_job(job.job_id, now=utcnow(), code="document_missing")
            return JobOutcome(job.job_id, job.doc_id, ok=False, error_code="document_missing")
        try:
            return self._process(job, document)
        except StateConflict as exc:
            # Somebody deleted or moved the document while it was queued.
            self._fail(job, document, "state_conflict", f"document is {exc.current}")
            return JobOutcome(job.job_id, job.doc_id, ok=False, error_code="state_conflict")
        except Exception as exc:  # a worker must never die of one bad file
            self._fail(job, document, "worker_crashed", str(exc)[:300])
            return JobOutcome(job.job_id, job.doc_id, ok=False, error_code="worker_crashed")

    def _fail(self, job: Job, document: Document, code: str, detail: str = "") -> None:
        now = utcnow()
        self.registry.fail_job(job.job_id, now=now, code=code, detail=detail)
        try:
            self.registry.update_document(
                document.doc_id, now=now, state=STATE_FAILED, error_code=code, error_detail=detail
            )
        except Exception:
            pass
        if document.blob_key:
            # A file we could not read is not worth keeping, and the uploader is
            # told why. Bytes we cannot use are bytes we should not store.
            self.blobs.delete(document.blob_key)
            try:
                self.registry.update_document(document.doc_id, now=now, blob_key=None)
            except Exception:
                pass

    def _process(self, job: Job, document: Document) -> JobOutcome:
        now = utcnow()
        # A retry is legitimate from any live state: failed (user pressed retry),
        # indexed (re-index), extracting (the process died mid-job and the
        # startup requeue handed it back).
        self.registry.update_document(
            document.doc_id,
            now=now,
            expect=[STATE_UPLOADED, STATE_FAILED, STATE_INDEXED, STATE_EXTRACTING],
            state=STATE_EXTRACTING,
        )
        self.registry.progress_job(
            job.job_id, stage="extracting", progress=10, message="reading the PDF", now=now
        )

        # Retry safety: drop anything a previous attempt wrote before writing again.
        self.store.delete_by_source(document.source)

        extracted = self._extract(document)
        if not extracted.get("ok"):
            self._fail(
                job, document, extracted.get("error_code", "extract_failed"),
                extracted.get("error_detail", ""),
            )
            return JobOutcome(job.job_id, document.doc_id, ok=False,
                              error_code=extracted.get("error_code"))

        pages = extracted["pages"]
        page_count = extracted["page_count"]
        with_text = extracted["pages_with_text"]
        if page_count and with_text / page_count < MIN_TEXT_PAGE_RATIO:
            self._fail(
                job, document, "no_text_layer",
                f"only {with_text} of {page_count} pages carry text - run OCR first",
            )
            return JobOutcome(job.job_id, document.doc_id, ok=False, error_code="no_text_layer")

        injection = extracted["injection"]
        quarantined = bool(injection["hard"])

        self.registry.progress_job(
            job.job_id, stage="chunking", progress=40, message="splitting pages", now=utcnow()
        )
        records = self._records(document, pages, quarantined=quarantined)
        if not records:
            self._fail(job, document, "no_text_layer", "no usable text after chunking")
            return JobOutcome(job.job_id, document.doc_id, ok=False, error_code="no_text_layer")
        if len(records) > MAX_CHUNKS:
            self._fail(job, document, "rejected_oversize_text", f"{len(records)} chunks")
            return JobOutcome(job.job_id, document.doc_id, ok=False,
                              error_code="rejected_oversize_text")

        self.registry.progress_job(
            job.job_id, stage="embedding", progress=60,
            message=f"embedding {len(records)} chunks", now=utcnow(),
        )
        vectors = self.embedder.embed_documents([r.text for r in records])
        self.store.add(records, vectors)

        now = utcnow()
        self.registry.update_document(
            document.doc_id,
            now=now,
            state=STATE_INDEXED,
            pages=page_count,
            chunk_count=len(records),
            visibility="quarantined" if quarantined else "private",
            injection_score=int(injection["score"]),
            injection_hard=quarantined,
            error_code=None,
            error_detail=None,
        )
        self.registry.finish_job(job.job_id, now=now)
        return JobOutcome(
            job.job_id, document.doc_id, ok=True, chunks=len(records), quarantined=quarantined
        )

    # --- helpers --------------------------------------------------------
    def _extract(self, document: Document) -> dict[str, Any]:
        if not document.blob_key or not self.blobs.exists(document.blob_key):
            return {"ok": False, "error_code": "blob_missing", "error_detail": "file is gone"}
        blob = self.blobs.path_for(document.blob_key)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "extract.json"
            command = [
                sys.executable, "-m", "src.extract_worker", str(blob), str(out),
                "--max-pages", str(self.settings.upload_max_pages),
            ]
            try:
                subprocess.run(
                    command,
                    cwd=str(Path(__file__).resolve().parents[1]),
                    timeout=EXTRACT_TIMEOUT_S,
                    capture_output=True,
                )
            except subprocess.TimeoutExpired:
                # run() kills the child for us; a thread timeout could not have.
                return {"ok": False, "error_code": "extract_timeout",
                        "error_detail": f"gave up after {EXTRACT_TIMEOUT_S}s"}
            if not out.exists():
                return {"ok": False, "error_code": "extract_crashed",
                        "error_detail": "the extractor wrote nothing"}
            return json.loads(out.read_text(encoding="utf-8"))

    def _records(
        self, document: Document, pages: list[dict[str, Any]], *, quarantined: bool
    ) -> list[ChunkRecord]:
        tokenizer = get_tokenizer(
            self.settings.embed_model if self.settings.embed_backend != "hash" else None
        )
        visibility = "quarantined" if quarantined else "private"
        records: list[ChunkRecord] = []
        for page in pages:
            # A page that itself trips a hard rule is never embedded at all.
            if page["injection"]["hard"]:
                continue
            for chunk in chunk_text(
                page["text"],
                max_tokens=self.settings.chunk_tokens,
                overlap=self.settings.chunk_overlap,
                tokenizer=tokenizer,
                min_chars=self.settings.min_chunk_chars,
            ):
                records.append(
                    ChunkRecord(
                        id=f"{document.source}#p{page['page']}#c{chunk.index}",
                        text=chunk.text,
                        metadata={
                            "file": document.display_name,
                            "source": document.source,
                            "subject": document.subject,
                            "page": page["page"],
                            "chunk_index": chunk.index,
                            "file_hash": document.sha256,
                            "chars": len(chunk.text),
                            "visibility": visibility,
                            # Always a real string: Chroma drops a None value,
                            # and an absent owner_id makes the row unreachable
                            # by the private branch and by takedown.
                            "owner_id": document.owner_id,
                        },
                    )
                )
        return records


def enqueue_pending(registry: SqliteRegistry) -> int:
    """Count of jobs waiting - used by the intake valve."""
    return registry.queue_depth()


__all__ = ["IngestWorker", "JobOutcome", "enqueue_pending", "JOB_QUEUED"]
