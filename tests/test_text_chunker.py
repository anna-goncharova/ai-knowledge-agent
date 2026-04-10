import pytest

from src.text_chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_text


def test_short_text_returns_single_chunk():
    text = "This is a short document that fits into a single chunk."
    chunks = chunk_text(text)
    assert chunks == [text]


def test_long_text_respects_chunk_size_and_overlap():
    # Build a ~2500-char text with clear word boundaries.
    text = " ".join([f"word{i:04d}" for i in range(300)])
    chunks = chunk_text(text)

    assert len(chunks) > 1
    assert all(len(c) <= DEFAULT_CHUNK_SIZE for c in chunks)

    # Consecutive chunks should share some content (overlap).
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev[-DEFAULT_CHUNK_OVERLAP:]
        assert any(token and token in nxt for token in tail.split())


def test_custom_chunk_size_and_overlap():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=50, chunk_overlap=50)


def test_default_parameters():
    assert DEFAULT_CHUNK_SIZE == 500
    assert DEFAULT_CHUNK_OVERLAP == 50
