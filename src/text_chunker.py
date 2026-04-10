"""Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping chunks.

    Uses :class:`RecursiveCharacterTextSplitter` which tries to split on
    natural boundaries (paragraphs, sentences, words) before falling back
    to characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)
