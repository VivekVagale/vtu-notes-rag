"""HTTP service for the web front end.

    uvicorn src.api:app --port 8000

This is the only process that ever sees an LLM key or the vector index. The
browser talks to these endpoints and nothing else.

Library tiers (see README): curated = vetted notes, private = an uploader's own
files, community = contributed files that have been approved. Upload and
moderation endpoints are not here yet - this service is read-only.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import api_uploads
from .answer import RagEngine
from .api_uploads import principal, router as uploads_router
from .auth import Principal, get_auth
from .blobs import get_blob_store
from .config import load_settings
from .jobs import IngestWorker
from .llm import LLMError
from .registry import get_registry

SCOPES = ("curated", "private", "community")
PUBLIC_SCOPES = ["curated", "community"]

RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "20"))  # requests
RATE_WINDOW = int(os.getenv("API_RATE_WINDOW", "60"))  # seconds
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()
]

_hits: dict[str, deque[float]] = defaultdict(deque)
_engine: RagEngine | None = None
_worker: IngestWorker | None = None


def rate_limit(request: Request) -> None:
    """In-memory per-IP limit. Good enough until auth lands, not multi-instance."""
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _hits[client]
    while window and now - window[0] > RATE_WINDOW:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(429, f"Rate limit: {RATE_LIMIT} requests per {RATE_WINDOW}s")
    window.append(now)


def engine() -> RagEngine:
    if _engine is None:  # pragma: no cover - set during lifespan
        raise HTTPException(503, "engine not ready")
    return _engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _engine, _worker
    settings = load_settings()
    _engine = RagEngine(settings)
    # Warm the embedder so the first real question is not the slow one.
    _engine.retriever.embedder.embed_query("warmup")

    registry = get_registry(settings)
    blobs = get_blob_store(settings)
    _worker = IngestWorker(
        settings,
        store=_engine.retriever.store,
        embedder=_engine.retriever.embedder,
        blobs=blobs,
    )
    api_uploads.deps.registry = registry
    api_uploads.deps.blobs = blobs
    api_uploads.deps.auth = get_auth(settings)
    api_uploads.deps.store = _engine.retriever.store
    api_uploads.deps.settings = settings
    api_uploads.deps.worker = _worker
    _worker.start()
    yield
    _worker.stop()
    registry.close()
    _worker = None
    _engine = None


app = FastAPI(
    title="VTU Notes RAG",
    version="0.1.0",
    description="Cited question answering over indexed VTU notes.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    # Without this the browser cannot read Location on the 202 and the
    # client has no way to find the job it just created.
    expose_headers=["Location", "Retry-After"],
)
app.include_router(uploads_router)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    subject: str | None = None
    scopes: list[Literal["curated", "private", "community"]] | None = Field(
        default=None, min_length=1, description="library tiers to search; omit for the public tiers"
    )
    k: int | None = Field(default=None, ge=1, le=12)
    exam_mode: bool = False
    marks: Literal[5, 10] = 10


@app.get("/health")
def health() -> dict[str, Any]:
    eng = engine()
    return {
        "status": "ok",
        "chunks": eng.retriever.count(),
        "subjects": eng.subjects(),
        "provider": eng.provider_name,
        "embed_backend": eng.settings.embed_backend,
        "reranker": eng.settings.reranker,
        "queue_depth": api_uploads.deps.registry.queue_depth()
        if api_uploads.deps.registry
        else 0,
        "disk_free_mb": api_uploads.deps.blobs.free_mb() if api_uploads.deps.blobs else None,
    }


@app.get("/subjects")
def subjects() -> dict[str, Any]:
    return {"subjects": engine().subjects()}


@app.post("/ask")
def ask(
    payload: AskRequest, request: Request, who: Principal = Depends(principal)
) -> dict[str, Any]:
    rate_limit(request)
    if payload.scopes and "private" in payload.scopes and who.is_anonymous:
        raise HTTPException(401, "auth_required_for_private")
    eng = engine()
    if eng.retriever.count() == 0:
        raise HTTPException(503, "Index is empty - run ingestion first")
    try:
        answer = eng.ask(
            payload.question,
            subject=payload.subject,
            k=payload.k,
            exam_mode=payload.exam_mode,
            marks=payload.marks,
            scopes=payload.scopes or PUBLIC_SCOPES,
            owner_id=who.user_id,
        )
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return answer.to_dict()
