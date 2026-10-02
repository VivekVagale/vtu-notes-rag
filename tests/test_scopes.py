"""Tier isolation on the read path.

These are the tests that have to hold before uploads exist, because every one of
them is a path by which one person's private notes could reach another person.
"""

from __future__ import annotations

import pytest

from src.embeddings import HashEmbedder
from src.ingest import ingest
from src.rerank import trust_of
from src.retrieve import Retriever
from src.store import ChunkRecord, VectorStore, build_scope_where

PRIVATE_TEXT = (
    "Alice private revision note: the deadlock detection algorithm runs every "
    "ten seconds in our lab assignment and logs the wait-for graph."
)


def _no_degenerate_operators(clause):
    """Chroma 1.5.9 raises on a one-element $and/$or and on an empty $in."""
    if isinstance(clause, dict):
        for key, value in clause.items():
            if key in {"$and", "$or"}:
                assert isinstance(value, list) and len(value) >= 2, f"{key} needs 2+ branches"
                for branch in value:
                    _no_degenerate_operators(branch)
            elif key == "$in":
                assert isinstance(value, list) and value, "$in must be non-empty"
            else:
                _no_degenerate_operators(value)


# --------------------------- clause construction ---------------------------


def test_single_tier_uses_equality_not_in():
    assert build_scope_where(tiers=["curated"]) == {"visibility": "curated"}


def test_two_tiers_use_in():
    clause = build_scope_where(tiers=["curated", "community"])
    assert clause == {"visibility": {"$in": ["curated", "community"]}}


def test_subject_and_tiers_compose():
    clause = build_scope_where(subject="OS", tiers=["curated", "community"])
    assert clause["$and"][0] == {"subject": "OS"}
    _no_degenerate_operators(clause)


def test_private_requires_an_owner():
    with pytest.raises(ValueError, match="owner"):
        build_scope_where(tiers=["private"])
    with pytest.raises(ValueError, match="owner"):
        build_scope_where(tiers=["curated", "private"], owner_id=None)


def test_private_is_reached_by_a_positive_owner_predicate():
    clause = build_scope_where(tiers=["private"], owner_id="alice")
    assert clause == {"$and": [{"visibility": "private"}, {"owner_id": "alice"}]}
    assert "$ne" not in repr(clause), "a negative predicate matches rows missing the key"


def test_mixed_tiers_or_two_branches():
    clause = build_scope_where(tiers=["curated", "private"], owner_id="alice")
    assert len(clause["$or"]) == 2
    _no_degenerate_operators(clause)


def test_empty_and_unknown_tiers_raise():
    with pytest.raises(ValueError):
        build_scope_where(tiers=[])
    with pytest.raises(ValueError, match="unknown"):
        build_scope_where(tiers=["public"])


def test_every_generated_clause_is_accepted_by_chroma(mini_corpus):
    """The degenerate-operator bugs only surface against the real engine."""
    store = VectorStore(mini_corpus)
    ingest(mini_corpus, store=store, embedder=HashEmbedder())
    embedder = HashEmbedder()
    vector = embedder.embed_query("locking")

    for kwargs in (
        {"tiers": ["curated"]},
        {"tiers": ["curated", "community"]},
        {"tiers": ["curated", "community", "private"], "owner_id": "alice"},
        {"tiers": ["private"], "owner_id": "alice"},
        {"subject": "DBMS", "tiers": ["curated"]},
        {"subject": "DBMS", "tiers": ["curated", "private"], "owner_id": "alice"},
    ):
        clause = build_scope_where(**kwargs)
        _no_degenerate_operators(clause)
        store.query(vector, 3, clause)  # must not raise


# --------------------------- retrieval behaviour ---------------------------


@pytest.fixture
def two_tenant_index(mini_corpus):
    """Curated notes, plus one private chunk owned by alice."""
    store = VectorStore(mini_corpus)
    ingest(mini_corpus, store=store, embedder=HashEmbedder())
    embedder = HashEmbedder()
    record = ChunkRecord(
        id="u/alice-doc#p1#c0",
        text=PRIVATE_TEXT,
        metadata={
            "file": "Alice_Notes.pdf",
            "source": "u/alicedoc",
            "subject": "OS",
            "page": 1,
            "chunk_index": 0,
            "visibility": "private",
            "owner_id": "alice",
        },
    )
    store.add([record], embedder.embed_documents([record.text]))
    return Retriever(mini_corpus, store=store, embedder=HashEmbedder())


def test_default_scope_never_returns_private_chunks(two_tenant_index):
    sources = two_tenant_index.retrieve("deadlock detection wait-for graph", k=5)
    assert sources, "the curated corpus should still answer"
    assert all(s.visibility != "private" for s in sources)
    assert all(s.file != "Alice_Notes.pdf" for s in sources)


def test_empty_scope_returns_nothing_rather_than_everything(two_tenant_index):
    assert two_tenant_index.retrieve("deadlock detection", scopes=[]) == []


def test_owner_sees_their_own_private_chunk(two_tenant_index):
    sources = two_tenant_index.retrieve(
        "deadlock detection wait-for graph", scopes=["private"], owner_id="alice", k=5
    )
    assert [s.file for s in sources] == ["Alice_Notes.pdf"]
    assert sources[0].visibility == "private"


def test_another_user_cannot_see_it(two_tenant_index):
    assert (
        two_tenant_index.retrieve(
            "deadlock detection wait-for graph", scopes=["private"], owner_id="bob", k=5
        )
        == []
    )


def test_private_without_an_owner_is_refused_not_served(two_tenant_index):
    with pytest.raises(ValueError, match="owner"):
        two_tenant_index.retrieve("deadlock detection", scopes=["private"])


# --------------------------- ranking ---------------------------


def test_unknown_tier_scores_zero_so_a_filter_mistake_cannot_promote_it():
    assert trust_of({"metadata": {"visibility": "curated"}}) == 1.0
    assert trust_of({"metadata": {"visibility": "community"}}) == 0.94
    assert trust_of({"metadata": {"visibility": "quarantined"}}) == 0.0
    assert trust_of({"metadata": {}}) == 1.0  # legacy chunks predate the tag


# --------------------------- publishing a document ---------------------------


def test_set_visibility_moves_every_chunk_and_reports_the_count(two_tenant_index):
    store = two_tenant_index.store
    assert store.count_by_source("u/alicedoc") == 1

    moved = store.set_visibility("u/alicedoc", "community")
    assert moved == 1


def test_set_visibility_merges_metadata_rather_than_replacing_it(two_tenant_index):
    """owner_id must survive publication or the document becomes un-takedownable."""
    store = two_tenant_index.store
    chunk_id = store.chunk_ids_for_source("u/alicedoc")[0]
    store.set_visibility("u/alicedoc", "community")

    row = store.collection.get(ids=[chunk_id], include=["metadatas", "documents"])
    meta = row["metadatas"][0]
    assert meta["visibility"] == "community"
    assert meta["owner_id"] == "alice", "ownership lost on publish"
    assert meta["page"] == 1 and meta["subject"] == "OS"
    assert row["documents"][0] == PRIVATE_TEXT, "chunk text must not be rewritten"


def test_publishing_flips_who_can_find_it(two_tenant_index):
    question = "deadlock detection wait-for graph"
    assert two_tenant_index.retrieve(question, k=5) == [] or all(
        s.file != "Alice_Notes.pdf" for s in two_tenant_index.retrieve(question, k=5)
    )

    two_tenant_index.store.set_visibility("u/alicedoc", "community")

    public = two_tenant_index.retrieve(question, k=5)
    assert any(s.file == "Alice_Notes.pdf" for s in public), "approved doc must be public"
    assert all(s.visibility != "private" for s in public)

    still_private = two_tenant_index.retrieve(
        question, scopes=["private"], owner_id="alice", k=5
    )
    assert still_private == [], "it left the private tier"


def test_set_visibility_on_an_unknown_source_reports_zero(two_tenant_index):
    """A silent no-op here would make Approve report success and publish nothing."""
    assert two_tenant_index.store.set_visibility("u/does-not-exist", "community") == 0
    assert two_tenant_index.store.count_by_source("u/does-not-exist") == 0
