"""The vertical slice, through HTTP only.

sign in -> upload -> indexed privately -> offer -> owner approves -> the public
can now get an answer from it. Plus the refusals that keep it honest.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src import api
from src.embeddings import HashEmbedder
from src.ingest import ingest
from src.store import VectorStore
from tests.conftest import MINI_PAGES, write_pdf

OWNER_EMAIL = "owner@example.com"


@pytest.fixture
def client(mini_corpus, tmp_path, monkeypatch):
    store = VectorStore(mini_corpus)
    ingest(mini_corpus, store=store, embedder=HashEmbedder())

    for key, value in {
        "PDF_DIR": str(mini_corpus.pdf_dir),
        "INDEX_DIR": str(mini_corpus.index_dir),
        "CHROMA_COLLECTION": mini_corpus.collection,
        "EMBED_BACKEND": "hash",
        "LLM_PROVIDER": "extractive",
        "RERANKER": "lexical",
        "REGISTRY_PATH": str(tmp_path / "registry.db"),
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "OWNER_EMAIL": OWNER_EMAIL,
        "AUTH_SECRET": "test-secret",
        "ALLOW_DEV_LOGIN": "true",
    }.items():
        monkeypatch.setenv(key, value)

    api._hits.clear()
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def pdf_bytes(tmp_path):
    path = write_pdf(tmp_path / "upload_me.pdf", MINI_PAGES)
    return path.read_bytes()


def login(client, email):
    response = client.post("/auth/dev-login", json={"email": email})
    assert response.status_code == 200, response.text
    return response.json()


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def upload(client, token, data, *, name="Module2.pdf", subject="DBMS"):
    return client.post(
        "/uploads",
        files={"file": (name, data, "application/pdf")},
        data={"subject": subject},
        headers=auth(token),
    )


def wait_for_job(client, token, job_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    last = {}
    while time.monotonic() < deadline:
        last = client.get(f"/jobs/{job_id}", headers=auth(token)).json()
        if last["state"] in {"done", "failed"}:
            return last
        time.sleep(0.2)
    raise AssertionError(f"job never finished: {last}")


# --------------------------- the happy path ---------------------------


def test_upload_stays_private_until_the_owner_approves_it(client, pdf_bytes):
    alice = login(client, "alice@example.com")
    assert alice["is_owner"] is False

    created = upload(client, alice["token"], pdf_bytes)
    assert created.status_code == 202, created.text
    assert created.headers["Location"].startswith("/jobs/")
    body = created.json()

    job = wait_for_job(client, alice["token"], body["job_id"])
    assert job["state"] == "done", job

    documents = client.get("/documents", headers=auth(alice["token"])).json()["documents"]
    assert len(documents) == 1
    document = documents[0]
    assert document["state"] == "indexed"
    assert document["visibility"] == "private"
    assert document["review_state"] is None
    assert document["chunk_count"] >= 3
    assert "owner_id" not in document and "blob_key" not in document

    question = {"question": "What is strict two phase locking?"}

    # 1. the public cannot reach it
    public = client.post("/ask", json=question).json()
    assert all(s["file"] != "Module2.pdf" for s in public["sources"])

    # 2. the uploader can
    mine = client.post(
        "/ask", json={**question, "scopes": ["private"]}, headers=auth(alice["token"])
    ).json()
    assert any(s["file"] == "Module2.pdf" for s in mine["sources"])
    assert all(s["visibility"] == "private" for s in mine["sources"])

    # 3. offering it needs an attestation
    doc_id = document["doc_id"]
    refused = client.put(
        f"/documents/{doc_id}/offer", json={"attested": False}, headers=auth(alice["token"])
    )
    assert refused.status_code == 400
    assert refused.json()["detail"]["code"] == "attestation_required"

    offered = client.put(
        f"/documents/{doc_id}/offer", json={"attested": True}, headers=auth(alice["token"])
    )
    assert offered.status_code == 200
    assert offered.json()["review_state"] == "pending"

    # 4. still not public while it waits
    waiting = client.post("/ask", json=question).json()
    assert all(s["file"] != "Module2.pdf" for s in waiting["sources"])

    # 5. the uploader is not a moderator
    assert client.get("/moderation/queue", headers=auth(alice["token"])).status_code == 403

    # 6. the owner approves
    owner = login(client, OWNER_EMAIL)
    assert owner["is_owner"] is True
    queue = client.get("/moderation/queue", headers=auth(owner["token"])).json()["queue"]
    assert [q["doc_id"] for q in queue] == [doc_id]
    assert queue[0]["display_name"] == "Module2.pdf" and queue[0]["attested"] is True

    approved = client.post(
        f"/moderation/{doc_id}/approve", json={"note": "looks like real notes"},
        headers=auth(owner["token"]),
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["chunks_published"] == document["chunk_count"]

    # 7. now anyone can get an answer out of it, with its citation
    published = client.post("/ask", json=question).json()
    contributed = [s for s in published["sources"] if s["file"] == "Module2.pdf"]
    assert contributed, published
    assert all(s["visibility"] == "community" for s in contributed)
    assert contributed[0]["citation"].startswith("[Module2.pdf, p.")


def test_withdrawing_an_approved_document_unpublishes_it(client, pdf_bytes):
    alice = login(client, "alice@example.com")
    body = upload(client, alice["token"], pdf_bytes).json()
    wait_for_job(client, alice["token"], body["job_id"])
    doc_id = body["doc_id"]

    client.put(f"/documents/{doc_id}/offer", json={"attested": True}, headers=auth(alice["token"]))
    owner = login(client, OWNER_EMAIL)
    client.post(f"/moderation/{doc_id}/approve", json={}, headers=auth(owner["token"]))

    question = {"question": "What is strict two phase locking?"}
    assert any(s["file"] == "Module2.pdf" for s in client.post("/ask", json=question).json()["sources"])

    withdrawn = client.delete(f"/documents/{doc_id}/offer", headers=auth(alice["token"]))
    assert withdrawn.status_code == 200
    after = client.post("/ask", json=question).json()
    assert all(s["file"] != "Module2.pdf" for s in after["sources"])


def test_rejection_leaves_the_uploaders_copy_alone(client, pdf_bytes):
    alice = login(client, "alice@example.com")
    body = upload(client, alice["token"], pdf_bytes).json()
    wait_for_job(client, alice["token"], body["job_id"])
    doc_id = body["doc_id"]
    client.put(f"/documents/{doc_id}/offer", json={"attested": True}, headers=auth(alice["token"]))

    owner = login(client, OWNER_EMAIL)
    rejected = client.post(
        f"/moderation/{doc_id}/reject", json={"note": "not your notes"},
        headers=auth(owner["token"]),
    )
    assert rejected.status_code == 200
    assert rejected.json()["review_state"] == "rejected"

    mine = client.post(
        "/ask",
        json={"question": "What is strict two phase locking?", "scopes": ["private"]},
        headers=auth(alice["token"]),
    ).json()
    assert any(s["file"] == "Module2.pdf" for s in mine["sources"]), "their own copy survives"


# --------------------------- refusals ---------------------------


def test_uploading_needs_an_account(client, pdf_bytes):
    response = client.post("/uploads", files={"file": ("x.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 401


def test_non_pdf_is_refused_on_its_bytes_not_its_name(client):
    alice = login(client, "alice@example.com")
    response = upload(client, alice["token"], b"GIF89a pretending to be a pdf", name="notes.pdf")
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "not_a_pdf"


def test_oversize_upload_is_refused(client, monkeypatch, pdf_bytes):
    alice = login(client, "alice@example.com")
    import src.api_uploads as uploads

    # Settings is frozen on purpose, so swap in a narrowed copy.
    monkeypatch.setattr(
        uploads.deps, "settings", uploads.deps.settings.with_overrides(upload_max_mb=0)
    )
    response = upload(client, alice["token"], pdf_bytes)
    assert response.status_code == 413


def test_the_same_file_twice_is_reported_as_a_duplicate(client, pdf_bytes):
    alice = login(client, "alice@example.com")
    first = upload(client, alice["token"], pdf_bytes)
    assert first.status_code == 202

    again = upload(client, alice["token"], pdf_bytes, name="renamed.pdf")
    assert again.status_code == 200
    assert again.json()["duplicate"] is True
    assert again.json()["doc_id"] == first.json()["doc_id"]


def test_one_user_cannot_read_another_users_job(client, pdf_bytes):
    alice = login(client, "alice@example.com")
    body = upload(client, alice["token"], pdf_bytes).json()

    bob = login(client, "bob@example.com")
    response = client.get(f"/jobs/{body['job_id']}", headers=auth(bob["token"]))
    assert response.status_code == 404, "404 not 403 - a 403 would confirm it exists"


def test_a_forged_token_is_rejected(client):
    alice = login(client, "alice@example.com")
    tampered = alice["token"][:-6] + "AAAAAA"
    assert client.get("/me", headers=auth(tampered)).status_code == 401


def test_moderation_is_closed_to_everyone_but_the_owner(client):
    alice = login(client, "alice@example.com")
    assert client.get("/moderation/queue").status_code in {401, 403}
    assert client.get("/moderation/queue", headers=auth(alice["token"])).status_code == 403


def test_me_reports_quota(client, pdf_bytes):
    alice = login(client, "alice@example.com")
    upload(client, alice["token"], pdf_bytes)
    body = client.get("/me", headers=auth(alice["token"])).json()
    assert body["documents"] == 1
    assert body["bytes_used"] > 0
    assert body["quota_documents"] >= 1


def test_health_reports_the_intake_valves(client):
    body = client.get("/health").json()
    assert body["queue_depth"] == 0
    assert body["disk_free_mb"] > 0
