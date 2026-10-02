"""Service tests. Offline: hash embeddings, extractive answers, throwaway index."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import api
from src.config import NOT_FOUND_MESSAGE
from src.embeddings import HashEmbedder
from src.ingest import ingest
from src.store import VectorStore


@pytest.fixture
def client(mini_corpus, monkeypatch):
    """A TestClient wired to a temporary corpus, not the real data/index."""
    store = VectorStore(mini_corpus)
    ingest(mini_corpus, store=store, embedder=HashEmbedder())

    for key, value in {
        "PDF_DIR": str(mini_corpus.pdf_dir),
        "INDEX_DIR": str(mini_corpus.index_dir),
        "CHROMA_COLLECTION": mini_corpus.collection,
        "EMBED_BACKEND": "hash",
        "LLM_PROVIDER": "extractive",
        "RERANKER": "lexical",
    }.items():
        monkeypatch.setenv(key, value)

    api._hits.clear()
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_the_live_index(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert body["embed_backend"] == "hash"
    assert body["provider"] == "extractive"
    assert "DBMS" in body["subjects"]


def test_subjects_endpoint(client):
    assert client.get("/subjects").json()["subjects"] == ["DBMS"]


def test_ask_returns_a_cited_answer_with_provenance(client):
    response = client.post("/ask", json={"question": "What is strict two phase locking?"})
    assert response.status_code == 200
    body = response.json()

    assert body["not_found"] is False
    assert body["sources"], "an answered question must carry its sources"
    assert all(s["visibility"] == "curated" for s in body["sources"])
    assert all(s["citation"].startswith("[") for s in body["sources"])
    cited_pages = {s["page"] for s in body["sources"]}
    assert cited_pages <= {1, 2, 3}
    assert body["warnings"] == []


def test_ask_honours_the_subject_filter(client):
    body = client.post(
        "/ask", json={"question": "What is write ahead logging?", "subject": "DBMS"}
    ).json()
    assert all(s["subject"] == "DBMS" for s in body["sources"])


def test_ask_scoped_to_community_finds_nothing_in_a_curated_corpus(client):
    """Tier isolation: a curated-only index must not answer a community-scoped ask."""
    body = client.post(
        "/ask", json={"question": "What is strict two phase locking?", "scopes": ["community"]}
    ).json()
    assert body["sources"] == []
    assert body["answer"] == NOT_FOUND_MESSAGE
    assert body["not_found"] is True


def test_ask_refuses_malformed_requests(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "a" * 501}).status_code == 422
    assert client.post("/ask", json={"question": "what is logging", "k": 99}).status_code == 422
    assert (
        client.post("/ask", json={"question": "what is logging", "scopes": ["public"]}).status_code
        == 422
    )
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": "what is logging", "scopes": []}).status_code == 422


def test_private_scope_is_refused_while_there_is_no_auth(client):
    """Without an identity there is nobody who could own private chunks."""
    response = client.post(
        "/ask", json={"question": "what is logging", "scopes": ["private"]}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "auth_required_for_private"


def test_answer_reports_the_tiers_actually_searched(client):
    body = client.post("/ask", json={"question": "What is two phase locking?"}).json()
    assert body["scopes"] == ["curated", "community"]
    assert all(s["visibility"] in {"curated", "community"} for s in body["sources"])


def test_rate_limit_rejects_the_overflow(client, monkeypatch):
    monkeypatch.setattr(api, "RATE_LIMIT", 3)
    api._hits.clear()
    payload = {"question": "What is checkpointing?"}
    codes = [client.post("/ask", json=payload).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]


def test_exam_mode_through_the_api(client):
    body = client.post(
        "/ask", json={"question": "Explain write ahead logging.", "exam_mode": True, "marks": 5}
    ).json()
    for heading in ("Definition", "Key points", "Diagram suggestion", "Conclusion"):
        assert heading in body["answer"]
    assert body["marks"] == 5
