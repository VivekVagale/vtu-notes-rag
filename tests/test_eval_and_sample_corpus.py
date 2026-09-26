"""Eval harness tests, plus an end-to-end run over the generated sample PDFs."""

from __future__ import annotations

import pytest

from src.config import PROJECT_ROOT, load_settings
from src.embeddings import HashEmbedder
from src.evaluate import QuestionSpec, evaluate, load_questions
from src.ingest import ingest
from src.retrieve import Retriever
from src.store import VectorStore

SAMPLE_DIR = load_settings().pdf_dir
SAMPLE_PDFS = [
    SAMPLE_DIR / "DBMS" / "DBMS_Module2.pdf",
    SAMPLE_DIR / "OS" / "OS_Module3.pdf",
]


def test_question_file_parses_and_is_well_formed():
    questions = load_questions(PROJECT_ROOT / "eval" / "questions.json")
    assert len(questions) >= 10
    assert all(q.expected_file.endswith(".pdf") for q in questions)
    assert all(q.expected_pages for q in questions)
    assert len({q.id for q in questions}) == len(questions)


def test_evaluate_scores_hits_and_misses(mini_corpus):
    store = VectorStore(mini_corpus)
    ingest(mini_corpus, store=store, embedder=HashEmbedder())
    retriever = Retriever(mini_corpus, store=store, embedder=HashEmbedder())

    questions = [
        QuestionSpec(
            id="hit",
            question="What is write ahead logging?",
            expected_file="DBMS_Mini.pdf",
            expected_pages=[3],
            subject="DBMS",
        ),
        QuestionSpec(
            id="miss",
            question="What is write ahead logging?",
            expected_file="DBMS_Mini.pdf",
            expected_pages=[99],
            subject="DBMS",
        ),
    ]
    report = evaluate(retriever, questions, k=3)
    assert report["hit_rate"] == 0.5
    assert report["results"][0].hit and report["results"][0].rank == 1
    assert not report["results"][1].hit
    assert 0 < report["mrr"] <= 1


@pytest.mark.skipif(
    not all(p.exists() for p in SAMPLE_PDFS),
    reason="run: python scripts/make_sample_pdfs.py",
)
def test_sample_corpus_retrieval_hit_rate(tmp_path):
    """Index the real sample PDFs into a throwaway store and score the eval set."""
    settings = load_settings(
        index_dir=tmp_path / "index",
        collection="sample_eval",
        embed_backend="hash",
        top_k=5,
        fetch_k=20,
        reranker="lexical",
    )
    store = VectorStore(settings)
    stats = ingest(settings, store=store, embedder=HashEmbedder())
    assert stats.files_indexed == 2
    assert stats.pages == 14  # 8 DBMS pages + 6 OS pages

    retriever = Retriever(settings, store=store, embedder=HashEmbedder())
    report = evaluate(retriever, load_questions(), k=5)
    assert report["hit_rate"] >= 0.7, report["results"]
