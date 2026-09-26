from __future__ import annotations

import pytest

from src.chunking import WordTokenizer, chunk_text, get_tokenizer

TEXT = " ".join(f"word{i}" for i in range(400))


def test_chunks_respect_size_and_overlap():
    tok = WordTokenizer()
    chunks = chunk_text(TEXT, max_tokens=100, overlap=20, tokenizer=tok, min_chars=1)
    assert len(chunks) > 1
    window = tok.budget(100)
    assert all(c.n_units <= window for c in chunks)

    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    shared = set(first_words) & set(second_words)
    assert shared, "consecutive chunks must overlap"


def test_offsets_point_back_at_the_source_text():
    chunks = chunk_text(TEXT, max_tokens=60, overlap=10, tokenizer=WordTokenizer(), min_chars=1)
    for chunk in chunks:
        assert TEXT[chunk.char_start : chunk.char_end] == chunk.text


def test_chunk_indices_are_sequential():
    chunks = chunk_text(TEXT, max_tokens=50, overlap=10, tokenizer=WordTokenizer(), min_chars=1)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_and_blank_text_give_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_short_text_still_yields_one_chunk():
    chunks = chunk_text("ACID stands for atomicity.", max_tokens=100, overlap=20, min_chars=200)
    assert len(chunks) == 1


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError):
        chunk_text(TEXT, max_tokens=50, overlap=50)


def test_get_tokenizer_falls_back_when_model_is_unavailable():
    tok = get_tokenizer("definitely/not-a-real-model-name")
    assert isinstance(tok, WordTokenizer)
    assert tok.count("one two three") == 3
