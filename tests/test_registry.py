from __future__ import annotations

import pytest

from src.registry import (
    JOB_QUEUED,
    JOB_RUNNING,
    REVIEW_APPROVED,
    REVIEW_PENDING,
    STATE_INDEXED,
    STATE_UPLOADED,
    Document,
    DuplicateDocument,
    NotFound,
    SqliteRegistry,
    StateConflict,
    new_id,
    source_for,
    utcnow,
)

NOW = "2026-10-02T00:00:00Z"
LATER = "2026-10-02T00:05:00Z"


@pytest.fixture
def registry(tmp_path):
    reg = SqliteRegistry(tmp_path / "registry.db")
    reg.migrate()
    yield reg
    reg.close()


@pytest.fixture
def alice(registry):
    return registry.upsert_user(email="Alice@Example.COM", now=NOW)


def make_doc(owner_id: str, *, sha: str = "a" * 64, name: str = "Mod2.pdf") -> Document:
    doc_id = new_id()
    return Document(
        doc_id=doc_id,
        owner_id=owner_id,
        source=source_for(doc_id),
        display_name=name,
        subject="DBMS",
        sha256=sha,
        bytes=1024,
    )


def test_migrate_is_idempotent(tmp_path):
    reg = SqliteRegistry(tmp_path / "r.db")
    reg.migrate()
    reg.migrate()
    reg.close()


def test_emails_are_normalised_and_users_are_not_duplicated(registry):
    first = registry.upsert_user(email="Alice@Example.COM", now=NOW)
    second = registry.upsert_user(email="  alice@example.com ", now=LATER)
    assert first.user_id == second.user_id
    assert first.email_norm == "alice@example.com"


def test_owner_flag_can_be_granted_later(registry):
    user = registry.upsert_user(email="me@example.com", now=NOW)
    assert not user.is_owner
    promoted = registry.upsert_user(email="me@example.com", now=LATER, is_owner=True)
    assert promoted.is_owner and promoted.user_id == user.user_id


def test_source_carries_no_filename(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id, name="../../evil name.pdf"), now=NOW)
    assert doc.source == f"u/{doc.doc_id}"
    assert "evil" not in doc.source and "/" == doc.source[1]


def test_documents_start_private_and_unoffered(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    assert doc.visibility == "private"
    assert doc.review_state is None
    assert doc.state == STATE_UPLOADED


def test_same_file_twice_from_one_owner_is_a_duplicate(registry, alice):
    registry.create_document(make_doc(alice.user_id, sha="b" * 64), now=NOW)
    with pytest.raises(DuplicateDocument) as exc:
        registry.create_document(make_doc(alice.user_id, sha="b" * 64), now=LATER)
    assert exc.value.existing_id


def test_two_people_may_each_hold_the_same_file(registry, alice):
    bob = registry.upsert_user(email="bob@example.com", now=NOW)
    registry.create_document(make_doc(alice.user_id, sha="c" * 64), now=NOW)
    registry.create_document(make_doc(bob.user_id, sha="c" * 64), now=NOW)  # must not raise


def test_in_library_only_matches_published_copies(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id, sha="d" * 64), now=NOW)
    assert registry.in_library("d" * 64) is None
    registry.update_document(doc.doc_id, now=LATER, visibility="community")
    assert registry.in_library("d" * 64).doc_id == doc.doc_id


def test_conditional_update_rejects_a_stale_expectation(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    registry.update_document(doc.doc_id, now=LATER, expect=[STATE_UPLOADED], state=STATE_INDEXED)
    with pytest.raises(StateConflict) as exc:
        registry.update_document(
            doc.doc_id, now=LATER, expect=[STATE_UPLOADED], state=STATE_INDEXED
        )
    assert exc.value.current == STATE_INDEXED


def test_updating_a_missing_document_is_not_found(registry):
    with pytest.raises(NotFound):
        registry.update_document("nope", now=NOW, state=STATE_INDEXED)


def test_review_transitions_are_guarded(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    offered = registry.set_review_state(doc.doc_id, to=REVIEW_PENDING, now=NOW, expect=[None])
    assert offered.review_state == REVIEW_PENDING

    with pytest.raises(StateConflict):
        registry.set_review_state(doc.doc_id, to=REVIEW_PENDING, now=LATER, expect=[None])

    approved = registry.set_review_state(
        doc.doc_id, to=REVIEW_APPROVED, now=LATER, expect=[REVIEW_PENDING], reviewed_by="owner"
    )
    assert approved.review_state == REVIEW_APPROVED and approved.reviewed_by == "owner"


def test_moderation_queue_lists_only_pending(registry, alice):
    waiting = registry.create_document(make_doc(alice.user_id, sha="e" * 64), now=NOW)
    registry.create_document(make_doc(alice.user_id, sha="f" * 64), now=NOW)
    registry.set_review_state(waiting.doc_id, to=REVIEW_PENDING, now=NOW)

    queue = registry.list_documents(review_state=REVIEW_PENDING)
    assert [d.doc_id for d in queue] == [waiting.doc_id]


def test_documents_are_listed_per_owner(registry, alice):
    bob = registry.upsert_user(email="bob@example.com", now=NOW)
    registry.create_document(make_doc(alice.user_id, sha="1" * 64), now=NOW)
    registry.create_document(make_doc(bob.user_id, sha="2" * 64), now=NOW)
    assert len(registry.list_documents(owner_id=alice.user_id)) == 1
    assert len(registry.list_documents(owner_id=bob.user_id)) == 1


# --------------------------- jobs ---------------------------


def test_one_live_job_per_document(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    assert registry.enqueue_job(doc_id=doc.doc_id, now=NOW) is not None
    assert registry.enqueue_job(doc_id=doc.doc_id, now=NOW) is None, "double submit"

    job = registry.claim_job(now=NOW)
    registry.finish_job(job.job_id, now=LATER)
    assert registry.enqueue_job(doc_id=doc.doc_id, now=LATER) is not None, "retry after done"


def test_claim_is_exclusive(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    registry.enqueue_job(doc_id=doc.doc_id, now=NOW)

    first = registry.claim_job(now=NOW)
    second = registry.claim_job(now=NOW)
    assert first is not None and second is None
    assert first.state == JOB_RUNNING and first.attempts == 1


def test_progress_is_clamped_and_readable(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    job = registry.enqueue_job(doc_id=doc.doc_id, now=NOW)
    registry.progress_job(job.job_id, stage="embedding", progress=250, message="half", now=LATER)
    live = registry.get_job(job.job_id)
    assert live.progress == 100 and live.stage == "embedding" and live.message == "half"


def test_failure_records_a_code_the_client_can_act_on(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    job = registry.enqueue_job(doc_id=doc.doc_id, now=NOW)
    registry.fail_job(job.job_id, now=LATER, code="no_text_layer", detail="x" * 900)
    failed = registry.get_job(job.job_id)
    assert failed.error_code == "no_text_layer"
    assert len(failed.error_detail) <= 500


def test_jobs_running_at_a_crash_are_requeued(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    registry.enqueue_job(doc_id=doc.doc_id, now=NOW)
    registry.claim_job(now=NOW)
    assert registry.queue_depth() == 1

    assert registry.requeue_running_jobs(now=LATER) == 1
    assert registry.claim_job(now=LATER).state == JOB_RUNNING


def test_queue_depth_ignores_finished_work(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    job = registry.enqueue_job(doc_id=doc.doc_id, now=NOW)
    registry.finish_job(job.job_id, now=LATER)
    assert registry.queue_depth() == 0


def test_quota_counters(registry, alice):
    registry.create_document(make_doc(alice.user_id, sha="9" * 64), now=NOW)
    assert registry.count_documents(alice.user_id) == 1
    assert registry.bytes_used(alice.user_id) == 1024


def test_public_dict_hides_storage_and_owner(registry, alice):
    doc = registry.create_document(make_doc(alice.user_id), now=NOW)
    payload = doc.public_dict()
    assert "blob_key" not in payload and "owner_id" not in payload
    assert payload["doc_id"] == doc.doc_id


def test_utcnow_is_zulu():
    assert utcnow().endswith("Z")
