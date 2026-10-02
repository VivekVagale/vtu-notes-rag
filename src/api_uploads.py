"""Upload, job status, offer-to-library and moderation routes.

Kept out of api.py so the read-only service stays readable on its own. Mounted
by api.py via include_router.

Two rules run through all of it:
- 404, never 403, for somebody else's document. A 403 confirms the id exists.
- Publishing widens access, so SQL commits before Chroma: a crash in between
  leaves a document approved-but-not-yet-findable, which is recoverable.
  Narrowing does the opposite.
"""

from __future__ import annotations

from typing import Any, Iterator

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field

from .auth import ANONYMOUS, AuthError, LocalAuthDriver, Principal
from .blobs import LocalBlobStore, NotAPdf, TooLarge
from .registry import (
    REVIEW_APPROVED,
    REVIEW_PENDING,
    REVIEW_REJECTED,
    STATE_INDEXED,
    Document,
    DuplicateDocument,
    SqliteRegistry,
    new_id,
    source_for,
    utcnow,
)
from .sanitize import safe_filename

router = APIRouter()

READ_CHUNK = 1 << 20
MAX_QUEUE_DEPTH = 50
MIN_FREE_MB = 2048


def fail(status: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message, **extra})


# --- wiring (api.py sets these on startup) ---------------------------------
class Deps:
    registry: SqliteRegistry | None = None
    blobs: LocalBlobStore | None = None
    auth: LocalAuthDriver | None = None
    store: Any = None
    settings: Any = None
    worker: Any = None


deps = Deps()


def _registry() -> SqliteRegistry:
    if deps.registry is None:  # pragma: no cover
        raise fail(503, "not_ready", "the service is still starting")
    return deps.registry


def principal(authorization: str | None = Header(default=None)) -> Principal:
    if deps.auth is None:  # pragma: no cover
        return ANONYMOUS
    try:
        return deps.auth.principal_from_header(authorization)
    except AuthError as exc:
        raise fail(401, "invalid_token", str(exc)) from exc


def require_user(who: Principal = Depends(principal)) -> Principal:
    if who.is_anonymous:
        raise fail(401, "auth_required", "sign in first")
    return who


def require_owner(who: Principal = Depends(principal)) -> Principal:
    if who.is_anonymous or not who.is_owner:
        # Same answer either way: never confirm that moderation exists.
        raise fail(403, "not_the_owner", "only the library owner can do this")
    return who


def _own_document(doc_id: str, who: Principal) -> Document:
    document = _registry().get_document(doc_id)
    if document is None or document.owner_id != who.user_id:
        raise fail(404, "not_found", "no such document")
    return document


# --- sign-in ---------------------------------------------------------------
class DevLogin(BaseModel):
    email: str = Field(min_length=3, max_length=200)


@router.post("/auth/dev-login")
def dev_login(payload: DevLogin) -> dict[str, Any]:
    """Local development sign-in. Disabled by ALLOW_DEV_LOGIN=false."""
    if not deps.settings or not deps.settings.allow_dev_login:
        raise fail(404, "not_found", "no such route")
    registry = _registry()
    email = payload.email.strip().lower()
    is_owner = bool(deps.auth and deps.auth.is_owner_email(email))
    with registry.unit_of_work():
        user = registry.upsert_user(email=email, now=utcnow(), is_owner=is_owner)
    token = deps.auth.tokens.issue(user_id=user.user_id, email=user.email_norm)
    return {"token": token, "user_id": user.user_id, "is_owner": is_owner}


@router.get("/me")
def me(who: Principal = Depends(require_user)) -> dict[str, Any]:
    registry = _registry()
    return {
        "user_id": who.user_id,
        "email": who.email,
        "is_owner": who.is_owner,
        "documents": registry.count_documents(who.user_id),
        "bytes_used": registry.bytes_used(who.user_id),
        "quota_documents": deps.settings.quota_docs_total,
        "quota_bytes": deps.settings.quota_bytes_total,
    }


# --- upload ----------------------------------------------------------------
def _stream(upload: UploadFile) -> Iterator[bytes]:
    while True:
        block = upload.file.read(READ_CHUNK)
        if not block:
            return
        yield block


@router.post("/uploads", status_code=202)
def create_upload(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    subject: str = Form("General"),
    who: Principal = Depends(require_user),
) -> dict[str, Any]:
    registry, blobs, settings = _registry(), deps.blobs, deps.settings

    # Valves first - cheapest checks, and they protect the box itself.
    if registry.queue_depth() > MAX_QUEUE_DEPTH:
        raise fail(503, "intake_paused", "too much work queued, try later")
    if blobs.free_mb() < MIN_FREE_MB:
        raise fail(503, "intake_paused", "the server is low on disk")

    max_bytes = settings.upload_max_mb * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + 4096:
        raise fail(413, "too_large", f"limit is {settings.upload_max_mb}MB")

    if registry.count_documents(who.user_id) >= settings.quota_docs_total:
        raise fail(429, "quota_documents", "document quota reached")
    if registry.bytes_used(who.user_id) >= settings.quota_bytes_total:
        raise fail(429, "quota_bytes", "storage quota reached")

    doc_id = new_id()
    try:
        info = blobs.put_stream(doc_id=doc_id, stream=_stream(file), max_bytes=max_bytes)
    except TooLarge as exc:
        raise fail(413, "too_large", f"limit is {settings.upload_max_mb}MB") from exc
    except NotAPdf as exc:
        raise fail(415, "not_a_pdf", "only PDF files are accepted") from exc

    published = registry.in_library(info.sha256)
    if published:
        blobs.delete(info.key)
        raise fail(409, "already_in_library", "this file is already in the library",
                   doc_id=published.doc_id)

    display_name = safe_filename(file.filename or "", fallback=f"document-{doc_id[:8]}.pdf")
    document = Document(
        doc_id=doc_id,
        owner_id=who.user_id,
        source=source_for(doc_id),
        display_name=display_name,
        subject=(subject or "General").strip()[:60] or "General",
        sha256=info.sha256,
        bytes=info.bytes,
        blob_key=info.key,
    )

    now = utcnow()
    try:
        with registry.unit_of_work():
            registry.create_document(document, now=now)
            job = registry.enqueue_job(doc_id=doc_id, now=now)
    except DuplicateDocument as exc:
        blobs.delete(info.key)
        existing = registry.get_document(exc.existing_id)
        response.status_code = 200
        return {"doc_id": exc.existing_id, "duplicate": True,
                "state": existing.state if existing else "unknown"}

    response.headers["Location"] = f"/jobs/{job.job_id}"
    return {"doc_id": doc_id, "job_id": job.job_id, "state": document.state, "duplicate": False}


@router.get("/jobs/{job_id}")
def job_status(job_id: str, who: Principal = Depends(require_user)) -> dict[str, Any]:
    registry = _registry()
    job = registry.get_job(job_id)
    if job is None:
        raise fail(404, "not_found", "no such job")
    _own_document(job.doc_id, who)  # 404s if it is not theirs
    return job.public_dict()


@router.get("/documents")
def my_documents(who: Principal = Depends(require_user)) -> dict[str, Any]:
    documents = _registry().list_documents(owner_id=who.user_id)
    return {"documents": [d.public_dict() for d in documents]}


# --- offering to the library ----------------------------------------------
class OfferRequest(BaseModel):
    attested: bool = Field(description="the uploader confirms they may share this")


@router.put("/documents/{doc_id}/offer")
def offer_document(
    doc_id: str, payload: OfferRequest, who: Principal = Depends(require_user)
) -> dict[str, Any]:
    if not payload.attested:
        raise fail(400, "attestation_required",
                   "confirm you have the right to share this before offering it")
    registry = _registry()
    document = _own_document(doc_id, who)
    if document.state != STATE_INDEXED:
        raise fail(409, "not_indexed", "wait for indexing to finish", state=document.state)
    if document.injection_hard:
        raise fail(409, "quarantined", "this file was flagged and cannot be offered")
    if document.review_state == REVIEW_PENDING:
        return {"doc_id": doc_id, "review_state": REVIEW_PENDING}

    now = utcnow()
    with registry.unit_of_work():
        registry.update_document(doc_id, now=now, attested=True)
        updated = registry.set_review_state(doc_id, to=REVIEW_PENDING, now=now)
    return {"doc_id": doc_id, "review_state": updated.review_state}


@router.delete("/documents/{doc_id}/offer")
def withdraw_offer(doc_id: str, who: Principal = Depends(require_user)) -> dict[str, Any]:
    registry = _registry()
    document = _own_document(doc_id, who)
    if document.review_state == REVIEW_APPROVED:
        # Narrowing: Chroma first, so a crash leaves it hidden rather than public.
        deps.store.set_visibility(document.source, "private")
    with registry.unit_of_work():
        registry.update_document(doc_id, now=utcnow(), visibility="private")
        registry.set_review_state(doc_id, to=None, now=utcnow())
    return {"doc_id": doc_id, "review_state": None, "visibility": "private"}


# --- moderation ------------------------------------------------------------
@router.get("/moderation/queue")
def moderation_queue(_owner: Principal = Depends(require_owner)) -> dict[str, Any]:
    documents = _registry().list_documents(review_state=REVIEW_PENDING)
    return {
        "queue": [
            {
                "doc_id": d.doc_id,
                "display_name": d.display_name,
                "subject": d.subject,
                "pages": d.pages,
                "chunks": d.chunk_count,
                "bytes": d.bytes,
                "injection_score": d.injection_score,
                "attested": d.attested,
                "created_at": d.created_at,
            }
            for d in documents
        ]
    }


class ReviewRequest(BaseModel):
    note: str = Field(default="", max_length=500)


@router.post("/moderation/{doc_id}/approve")
def approve(
    doc_id: str, payload: ReviewRequest, owner: Principal = Depends(require_owner)
) -> dict[str, Any]:
    registry = _registry()
    document = registry.get_document(doc_id)
    if document is None:
        raise fail(404, "not_found", "no such document")
    if document.review_state != REVIEW_PENDING:
        raise fail(409, "not_pending", "this document is not awaiting review",
                   current=document.review_state)
    if document.injection_hard:
        raise fail(409, "quarantined", "flagged documents cannot be published")

    now = utcnow()
    # Widening: record the decision first, then publish the chunks.
    with registry.unit_of_work():
        registry.set_review_state(
            doc_id, to=REVIEW_APPROVED, now=now, expect=[REVIEW_PENDING],
            reviewed_by=owner.user_id, note=payload.note,
        )
        registry.update_document(doc_id, now=now, visibility="community")

    moved = deps.store.set_visibility(document.source, "community")
    if moved == 0:
        # update() on a missing id is a silent no-op, so zero means nothing was
        # published. Roll the decision back rather than report a false success.
        with registry.unit_of_work():
            registry.set_review_state(doc_id, to=REVIEW_PENDING, now=utcnow())
            registry.update_document(doc_id, now=utcnow(), visibility="private")
        raise fail(500, "publish_failed", "no chunks were published; the index may be stale")

    return {"doc_id": doc_id, "review_state": REVIEW_APPROVED, "chunks_published": moved}


@router.post("/moderation/{doc_id}/reject")
def reject(
    doc_id: str, payload: ReviewRequest, owner: Principal = Depends(require_owner)
) -> dict[str, Any]:
    registry = _registry()
    document = registry.get_document(doc_id)
    if document is None:
        raise fail(404, "not_found", "no such document")
    if document.review_state != REVIEW_PENDING:
        raise fail(409, "not_pending", "this document is not awaiting review",
                   current=document.review_state)
    now = utcnow()
    with registry.unit_of_work():
        registry.set_review_state(
            doc_id, to=REVIEW_REJECTED, now=now, expect=[REVIEW_PENDING],
            reviewed_by=owner.user_id, note=payload.note,
        )
    # Rejection only clears the offer. The uploader keeps their private copy.
    return {"doc_id": doc_id, "review_state": REVIEW_REJECTED}
