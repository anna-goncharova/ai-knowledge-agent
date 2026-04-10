"""Tests for src.embedder using a fake OpenAI client and in-memory ChromaDB."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import chromadb
import pytest

from src.embedder import DEFAULT_EMBEDDING_MODEL, Embedder


EMBED_DIM = 16


def _deterministic_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Turn text into a stable pseudo-embedding without calling OpenAI."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Use bytes to produce floats in [-1, 1].
    values = [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(dim)]
    return values


@dataclass
class _FakeEmbeddingItem:
    embedding: list[float]


@dataclass
class _FakeEmbeddingsResponse:
    data: list[_FakeEmbeddingItem]


class _FakeEmbeddingsResource:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, *, model: str, input: Sequence[str]):
        self.calls.append({"model": model, "input": list(input)})
        return _FakeEmbeddingsResponse(
            data=[_FakeEmbeddingItem(_deterministic_embedding(t)) for t in input]
        )


class _FakeOpenAIClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddingsResource()


@pytest.fixture
def collection(request):
    # Use a unique collection name per test to avoid state leaking across
    # tests when ChromaDB's EphemeralClient shares backing state.
    client = chromadb.EphemeralClient()
    name = f"test_embedder_{request.node.name}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name)


@pytest.fixture
def fake_client():
    return _FakeOpenAIClient()


def test_embeds_and_stores_ten_documents(fake_client, collection):
    docs = [f"Document number {i}: the quick brown fox jumps." for i in range(10)]
    embedder = Embedder(client=fake_client, collection=collection)

    ids = embedder.add_chunks(docs)

    # All 10 stored.
    assert len(ids) == 10
    assert collection.count() == 10

    # OpenAI client was called once with the right model and all 10 inputs.
    assert len(fake_client.embeddings.calls) == 1
    call = fake_client.embeddings.calls[0]
    assert call["model"] == DEFAULT_EMBEDDING_MODEL
    assert call["input"] == docs

    # Retrieval returns the same documents we stored.
    got = collection.get(ids=ids)
    assert sorted(got["ids"]) == sorted(ids)
    assert set(got["documents"]) == set(docs)


def test_query_returns_nearest_document(fake_client, collection):
    docs = [f"unique text chunk {i}" for i in range(10)]
    embedder = Embedder(client=fake_client, collection=collection)
    embedder.add_chunks(docs)

    # Query with the same embedding the fake client would return for doc #3.
    target = docs[3]
    query_emb = _deterministic_embedding(target)
    result = collection.query(query_embeddings=[query_emb], n_results=1)
    assert result["documents"][0][0] == target


def test_custom_ids_and_metadata(fake_client, collection):
    docs = [f"doc-{i}" for i in range(10)]
    ids = [f"id-{i}" for i in range(10)]
    metas = [{"source": f"file-{i}.md"} for i in range(10)]

    embedder = Embedder(client=fake_client, collection=collection)
    returned_ids = embedder.add_chunks(docs, ids=ids, metadatas=metas)

    assert returned_ids == ids
    got = collection.get(ids=["id-0", "id-9"])
    assert got["metadatas"][0] == {"source": "file-0.md"} or got["metadatas"][1] == {"source": "file-0.md"}


def test_length_mismatch_raises(fake_client, collection):
    embedder = Embedder(client=fake_client, collection=collection)
    with pytest.raises(ValueError):
        embedder.add_chunks(["a", "b"], ids=["only-one"])
    with pytest.raises(ValueError):
        embedder.add_chunks(["a", "b"], metadatas=[{"x": 1}])


def test_empty_input_is_noop(fake_client, collection):
    embedder = Embedder(client=fake_client, collection=collection)
    assert embedder.add_chunks([]) == []
    assert collection.count() == 0
    assert fake_client.embeddings.calls == []
