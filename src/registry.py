"""Document and job registry.

SQLite today so the whole upload flow runs on one machine with no cloud account.
Every statement here is plain SQL that Postgres also accepts, and callers only
ever touch the Registry methods - so the Supabase driver is a new class, not a
rewrite. The same shape as get_embedder / get_reranker / get_provider.

Times are passed in, never read from the clock inside a method, so tests are
deterministic and a retry writes the same row it would have written.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import Settings

# --- document lifecycle -------------------------------------------------
STATE_UPLOADED = "uploaded"
STATE_EXTRACTING = "extracting"
STATE_INDEXED = "indexed"
STATE_FAILED = "failed"
STATE_DELETED = "deleted"

# --- review lifecycle (None = never offered to the library) -------------
REVIEW_PENDING = "pending"
REVIEW_APPROVED = "approved"
REVIEW_REJECTED = "rejected"

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    email_norm  TEXT NOT NULL UNIQUE,
    is_owner    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    owner_id        TEXT NOT NULL REFERENCES users(user_id),
    source          TEXT NOT NULL UNIQUE,
    display_name    TEXT NOT NULL,
    subject         TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    bytes           INTEGER NOT NULL,
    pages           INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    state           TEXT NOT NULL,
    visibility      TEXT NOT NULL,
    review_state    TEXT,
    review_note     TEXT NOT NULL DEFAULT '',
    reviewed_by     TEXT,
    attested        INTEGER NOT NULL DEFAULT 0,
    injection_score INTEGER NOT NULL DEFAULT 0,
    injection_hard  INTEGER NOT NULL DEFAULT 0,
    blob_key        TEXT,
    error_code      TEXT,
    error_detail    TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    CONSTRAINT uq_owner_sha UNIQUE (owner_id, sha256)
);
CREATE INDEX IF NOT EXISTS ix_docs_owner ON documents(owner_id, created_at);
CREATE INDEX IF NOT EXISTS ix_docs_review ON documents(review_state, created_at);

CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id       TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id),
    state        TEXT NOT NULL,
    stage        TEXT NOT NULL DEFAULT 'queued',
    progress     INTEGER NOT NULL DEFAULT 0,
    message      TEXT NOT NULL DEFAULT '',
    attempts     INTEGER NOT NULL DEFAULT 0,
    error_code   TEXT,
    error_detail TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
-- One live job per document. Partial unique index: SQLite and Postgres both honour it.
CREATE UNIQUE INDEX IF NOT EXISTS uq_job_active
    ON ingest_jobs(doc_id) WHERE state IN ('queued', 'running');
"""


class RegistryError(RuntimeError):
    pass


class DuplicateDocument(RegistryError):
    def __init__(self, existing_id: str) -> None:
        super().__init__(f"document already exists: {existing_id}")
        self.existing_id = existing_id


class StateConflict(RegistryError):
    def __init__(self, current: str | None) -> None:
        super().__init__(f"unexpected current state: {current}")
        self.current = current


class NotFound(RegistryError):
    pass


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_id() -> str:
    return uuid.uuid4().hex


def source_for(doc_id: str) -> str:
    """Uploads get a synthetic source carrying no uploader-supplied text.

    It is the Chroma filter key and the SQL unique key at once, so
    delete_by_source(source) is exactly 'take this document down', and no
    filename ever reaches chunk ids (which are returned to every caller).
    """
    return f"u/{doc_id}"


@dataclass
class User:
    user_id: str
    email_norm: str
    is_owner: bool
    created_at: str


@dataclass
class Document:
    doc_id: str
    owner_id: str
    source: str
    display_name: str
    subject: str
    sha256: str
    bytes: int
    pages: int = 0
    chunk_count: int = 0
    state: str = STATE_UPLOADED
    visibility: str = "private"
    review_state: str | None = None
    review_note: str = ""
    reviewed_by: str | None = None
    attested: bool = False
    injection_score: int = 0
    injection_hard: bool = False
    blob_key: str | None = None
    error_code: str | None = None
    error_detail: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def public_dict(self) -> dict[str, Any]:
        """What an uploader may see about their own document."""
        data = asdict(self)
        for hidden in ("blob_key", "owner_id"):
            data.pop(hidden, None)
        return data


@dataclass
class Job:
    job_id: str
    doc_id: str
    state: str = JOB_QUEUED
    stage: str = "queued"
    progress: int = 0
    message: str = ""
    attempts: int = 0
    error_code: str | None = None
    error_detail: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_to(cls, row: sqlite3.Row):
    fields = {f for f in cls.__dataclass_fields__}
    data = {k: row[k] for k in row.keys() if k in fields}
    for flag in ("is_owner", "attested", "injection_hard"):
        if flag in data:
            data[flag] = bool(data[flag])
    return cls(**data)


class SqliteRegistry:
    """The only module that writes SQL. Everything else calls these methods."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    # --- lifecycle ------------------------------------------------------
    def migrate(self) -> None:
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def unit_of_work(self) -> Iterator[None]:
        """BEGIN IMMEDIATE: two concurrent uploads cannot both pass a check."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    # --- users ----------------------------------------------------------
    def upsert_user(self, *, email: str, now: str, is_owner: bool = False) -> User:
        email_norm = email.strip().lower()
        row = self._conn.execute(
            "SELECT * FROM users WHERE email_norm = ?", (email_norm,)
        ).fetchone()
        if row:
            if is_owner and not row["is_owner"]:
                self._conn.execute(
                    "UPDATE users SET is_owner = 1 WHERE user_id = ?", (row["user_id"],)
                )
                row = self._conn.execute(
                    "SELECT * FROM users WHERE user_id = ?", (row["user_id"],)
                ).fetchone()
            return _row_to(User, row)
        user = User(user_id=new_id(), email_norm=email_norm, is_owner=is_owner, created_at=now)
        self._conn.execute(
            "INSERT INTO users (user_id, email_norm, is_owner, created_at) VALUES (?, ?, ?, ?)",
            (user.user_id, user.email_norm, int(user.is_owner), user.created_at),
        )
        return user

    def get_user(self, user_id: str) -> User | None:
        row = self._conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return _row_to(User, row) if row else None

    # --- documents ------------------------------------------------------
    def create_document(self, doc: Document, *, now: str) -> Document:
        doc.created_at = doc.created_at or now
        doc.updated_at = now
        try:
            self._conn.execute(
                """INSERT INTO documents (
                    doc_id, owner_id, source, display_name, subject, sha256, bytes, pages,
                    chunk_count, state, visibility, review_state, review_note, reviewed_by,
                    attested, injection_score, injection_hard, blob_key, error_code,
                    error_detail, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc.doc_id, doc.owner_id, doc.source, doc.display_name, doc.subject,
                    doc.sha256, doc.bytes, doc.pages, doc.chunk_count, doc.state,
                    doc.visibility, doc.review_state, doc.review_note, doc.reviewed_by,
                    int(doc.attested), doc.injection_score, int(doc.injection_hard),
                    doc.blob_key, doc.error_code, doc.error_detail, doc.created_at,
                    doc.updated_at,
                ),
            )
        except sqlite3.IntegrityError as exc:
            existing = self._conn.execute(
                "SELECT doc_id FROM documents WHERE owner_id = ? AND sha256 = ?",
                (doc.owner_id, doc.sha256),
            ).fetchone()
            if existing:
                raise DuplicateDocument(existing["doc_id"]) from exc
            raise
        return doc

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return _row_to(Document, row) if row else None

    def get_document_by_source(self, source: str) -> Document | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE source = ?", (source,)
        ).fetchone()
        return _row_to(Document, row) if row else None

    def list_documents(
        self,
        *,
        owner_id: str | None = None,
        review_state: str | None = None,
        states: Sequence[str] | None = None,
        limit: int = 50,
    ) -> list[Document]:
        where, params = ["1=1"], []
        if owner_id:
            where.append("owner_id = ?")
            params.append(owner_id)
        if review_state:
            where.append("review_state = ?")
            params.append(review_state)
        if states:
            where.append(f"state IN ({','.join('?' * len(states))})")
            params.extend(states)
        rows = self._conn.execute(
            f"SELECT * FROM documents WHERE {' AND '.join(where)} "
            "ORDER BY created_at DESC, doc_id LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [_row_to(Document, r) for r in rows]

    def update_document(
        self, doc_id: str, *, now: str, expect: Sequence[str] | None = None, **fields: Any
    ) -> Document:
        """Conditional update. rowcount != 1 means somebody else moved it."""
        if not fields:
            raise ValueError("nothing to update")
        for flag in ("attested", "injection_hard"):
            if flag in fields:
                fields[flag] = int(bool(fields[flag]))
        sets = ", ".join(f"{k} = ?" for k in fields)
        params: list[Any] = [*fields.values(), now, doc_id]
        sql = f"UPDATE documents SET {sets}, updated_at = ? WHERE doc_id = ?"
        if expect is not None:
            sql += f" AND state IN ({','.join('?' * len(expect))})"
            params.extend(expect)
        cursor = self._conn.execute(sql, params)
        if cursor.rowcount != 1:
            current = self.get_document(doc_id)
            if current is None:
                raise NotFound(doc_id)
            raise StateConflict(current.state)
        return self.get_document(doc_id)  # type: ignore[return-value]

    def set_review_state(
        self,
        doc_id: str,
        *,
        to: str | None,
        now: str,
        expect: Sequence[str | None] = (),
        reviewed_by: str | None = None,
        note: str = "",
    ) -> Document:
        current = self.get_document(doc_id)
        if current is None:
            raise NotFound(doc_id)
        if expect and current.review_state not in expect:
            raise StateConflict(current.review_state)
        self._conn.execute(
            "UPDATE documents SET review_state = ?, reviewed_by = ?, review_note = ?, "
            "updated_at = ? WHERE doc_id = ?",
            (to, reviewed_by, note, now, doc_id),
        )
        return self.get_document(doc_id)  # type: ignore[return-value]

    def find_by_sha256(self, sha256: str, *, owner_id: str | None = None) -> list[Document]:
        if owner_id:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE sha256 = ? AND owner_id = ?", (sha256, owner_id)
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchall()
        return [_row_to(Document, r) for r in rows]

    def in_library(self, sha256: str) -> Document | None:
        """Is this exact file already public, under anyone's account?"""
        row = self._conn.execute(
            "SELECT * FROM documents WHERE sha256 = ? AND visibility = 'community' "
            "AND state != 'deleted' LIMIT 1",
            (sha256,),
        ).fetchone()
        return _row_to(Document, row) if row else None

    # --- jobs -----------------------------------------------------------
    def enqueue_job(self, *, doc_id: str, now: str) -> Job | None:
        """None when a live job for that document already exists."""
        job = Job(job_id=new_id(), doc_id=doc_id, created_at=now, updated_at=now)
        try:
            self._conn.execute(
                "INSERT INTO ingest_jobs (job_id, doc_id, state, stage, progress, message, "
                "attempts, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (job.job_id, job.doc_id, job.state, job.stage, 0, "", 0, now, now),
            )
        except sqlite3.IntegrityError:
            return None
        return job

    def claim_job(self, *, now: str) -> Job | None:
        """Atomically take the oldest queued job. Safe against a second worker."""
        row = self._conn.execute(
            "SELECT job_id FROM ingest_jobs WHERE state = ? ORDER BY created_at, job_id LIMIT 1",
            (JOB_QUEUED,),
        ).fetchone()
        if row is None:
            return None
        cursor = self._conn.execute(
            "UPDATE ingest_jobs SET state = ?, stage = 'claimed', attempts = attempts + 1, "
            "updated_at = ? WHERE job_id = ? AND state = ?",
            (JOB_RUNNING, now, row["job_id"], JOB_QUEUED),
        )
        if cursor.rowcount != 1:
            return None  # another worker got it first
        return self.get_job(row["job_id"])

    def get_job(self, job_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return _row_to(Job, row) if row else None

    def latest_job_for(self, doc_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT * FROM ingest_jobs WHERE doc_id = ? ORDER BY created_at DESC LIMIT 1",
            (doc_id,),
        ).fetchone()
        return _row_to(Job, row) if row else None

    def progress_job(
        self, job_id: str, *, stage: str, progress: int, message: str, now: str
    ) -> None:
        self._conn.execute(
            "UPDATE ingest_jobs SET stage = ?, progress = ?, message = ?, updated_at = ? "
            "WHERE job_id = ?",
            (stage, max(0, min(100, progress)), message, now, job_id),
        )

    def finish_job(self, job_id: str, *, now: str) -> None:
        self._conn.execute(
            "UPDATE ingest_jobs SET state = ?, stage = 'done', progress = 100, updated_at = ? "
            "WHERE job_id = ?",
            (JOB_DONE, now, job_id),
        )

    def fail_job(self, job_id: str, *, now: str, code: str, detail: str = "") -> None:
        self._conn.execute(
            "UPDATE ingest_jobs SET state = ?, stage = 'failed', error_code = ?, "
            "error_detail = ?, updated_at = ? WHERE job_id = ?",
            (JOB_FAILED, code, detail[:500], now, job_id),
        )

    def queue_depth(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM ingest_jobs WHERE state IN (?, ?)",
            (JOB_QUEUED, JOB_RUNNING),
        ).fetchone()
        return int(row["n"])

    def requeue_running_jobs(self, *, now: str) -> int:
        """Called at startup: a job that was running when the process died is nobody's."""
        cursor = self._conn.execute(
            "UPDATE ingest_jobs SET state = ?, stage = 'requeued', updated_at = ? "
            "WHERE state = ?",
            (JOB_QUEUED, now, JOB_RUNNING),
        )
        return cursor.rowcount

    def count_documents(self, owner_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM documents WHERE owner_id = ? AND state != ?",
            (owner_id, STATE_DELETED),
        ).fetchone()
        return int(row["n"])

    def bytes_used(self, owner_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(bytes), 0) AS n FROM documents "
            "WHERE owner_id = ? AND state != ?",
            (owner_id, STATE_DELETED),
        ).fetchone()
        return int(row["n"])


def get_registry(settings: Settings) -> SqliteRegistry:
    registry = SqliteRegistry(settings.registry_path)
    registry.migrate()
    return registry
