from __future__ import annotations

import pytest

from src.answer import CITATION_RE, RagEngine, validate_citations
from src.config import NOT_FOUND_MESSAGE
from src.embeddings import HashEmbedder
from src.ingest import ingest
from src.rerank import LexicalReranker
from src.retrieve import Retriever
from src.store import VectorStore
from tests.conftest import MINI_PAGES


@pytest.fixture
def retriever(mini_corpus) -> Retriever:
    store = VectorStore(mini_corpus)
    ingest(mini_corpus, store=store, embedder=HashEmbedder())
    return Retriever(
        mini_corpus, store=store, embedder=HashEmbedder(), reranker=LexicalReranker()
    )


@pytest.fixture
def engine(retriever) -> RagEngine:
    return RagEngine(retriever.settings, retriever=retriever, provider=None)


def test_retrieval_finds_the_right_page(retriever):
    sources = retriever.retrieve("What is strict two phase locking?")
    assert sources
    assert sources[0].page == 2
    assert sources[0].file == "DBMS_Mini.pdf"
    assert sources[0].citation == "[DBMS_Mini.pdf, p.2]"


def test_retrieval_respects_top_k(retriever):
    assert len(retriever.retrieve("logging", k=1)) == 1


def test_subject_filter_excludes_other_subjects(retriever, make_pdf):
    settings = retriever.settings
    make_pdf(settings.pdf_dir / "OS" / "OS_Mini.pdf", MINI_PAGES[:1])
    ingest(settings, store=retriever.store, embedder=HashEmbedder())

    assert all(s.subject == "OS" for s in retriever.retrieve("transaction", subject="OS"))
    assert all(s.subject == "DBMS" for s in retriever.retrieve("transaction", subject="DBMS"))


def test_empty_question_returns_nothing(retriever):
    assert retriever.retrieve("   ") == []


def test_answer_is_grounded_and_cited(engine):
    answer = engine.ask("What does atomicity mean in a transaction?")
    assert not answer.not_found
    assert CITATION_RE.search(answer.text), answer.text
    cited = {(f.lower(), int(p)) for f, p in CITATION_RE.findall(answer.text)}
    allowed = {(s.file.lower(), s.page) for s in answer.sources}
    assert cited <= allowed
    assert answer.warnings == []


def test_unrelated_question_says_not_found(engine):
    answer = engine.ask("Explain the Kalman filter used in drone navigation.")
    assert answer.not_found
    assert answer.text == NOT_FOUND_MESSAGE


def test_exam_mode_uses_the_vtu_structure(engine):
    answer = engine.ask("Explain write ahead logging.", exam_mode=True, marks=10)
    for heading in ("Definition", "Key points", "Diagram suggestion", "Conclusion"):
        assert heading in answer.text
    assert CITATION_RE.search(answer.text)


def test_validate_citations_flags_pages_that_were_not_retrieved(engine):
    answer = engine.ask("What is checkpointing?")
    good = answer.sources[0].citation
    assert validate_citations(f"Claim {good}", answer.sources) == []

    bogus = validate_citations("Claim [Ghost.pdf, p.99]", answer.sources)
    assert bogus and "Ghost.pdf" in bogus[0]
    assert validate_citations("Claim with no citation at all", answer.sources)


def test_answer_serialises_for_the_json_cli(engine):
    payload = engine.ask("What is two phase locking?").to_dict()
    assert payload["sources"] and payload["sources"][0]["citation"].startswith("[")
    assert payload["provider"] == "extractive"
