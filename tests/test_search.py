"""Tests for src.search using a fake OpenAI client and in-memory ChromaDB."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import chromadb
import pytest

from src.embedder import Embedder
from src.search import SemanticSearcher, _format_results, main

EMBED_DIM = 16


def _deterministic_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(dim)]


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
    client = chromadb.EphemeralClient()
    name = f"test_search_{request.node.name}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name)


@pytest.fixture
def fake_client():
    return _FakeOpenAIClient()


@pytest.fixture
def populated(fake_client, collection):
    docs = [
        "Python is a high-level programming language.",
        "FastAPI is a modern web framework for Python.",
        "ChromaDB stores vector embeddings.",
        "OpenAI provides embedding models.",
        "RAG combines retrieval and generation.",
        "Docker containers package applications.",
        "Kubernetes orchestrates container deployments.",
        "Markdown is a lightweight markup language.",
        "Git is a distributed version control system.",
        "Linux is an open-source operating system.",
    ]
    embedder = Embedder(client=fake_client, collection=collection)
    embedder.add_chunks(docs, metadatas=[{"idx": i} for i in range(len(docs))])
    return docs


def test_search_returns_top_5_with_scores(populated, fake_client, collection):
    searcher = SemanticSearcher(client=fake_client, collection=collection)
    results = searcher.search("Python web framework")

    assert len(results) == 5
    # Every result has a score in (0, 1].
    for r in results:
        assert 0.0 < r.score <= 1.0
        assert r.id
        assert r.document
    # Results are ordered by descending score.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_exact_match_is_first_result(populated, fake_client, collection):
    target = populated[2]  # "ChromaDB stores vector embeddings."
    searcher = SemanticSearcher(client=fake_client, collection=collection)
    results = searcher.search(target, top_k=3)

    assert len(results) == 3
    assert results[0].document == target
    # Exact match distance is ~0 so score should be ~1.0.
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_custom_top_k(populated, fake_client, collection):
    searcher = SemanticSearcher(client=fake_client, collection=collection)
    assert len(searcher.search("anything", top_k=3)) == 3
    assert len(searcher.search("anything", top_k=1)) == 1


def test_invalid_query_or_top_k(populated, fake_client, collection):
    searcher = SemanticSearcher(client=fake_client, collection=collection)
    with pytest.raises(ValueError):
        searcher.search("")
    with pytest.raises(ValueError):
        searcher.search("   ")
    with pytest.raises(ValueError):
        searcher.search("q", top_k=0)


def test_embedding_call_uses_configured_model(populated, fake_client, collection):
    searcher = SemanticSearcher(
        client=fake_client, collection=collection, model="custom-model"
    )
    searcher.search("hello world")
    assert fake_client.embeddings.calls[-1]["model"] == "custom-model"
    assert fake_client.embeddings.calls[-1]["input"] == ["hello world"]


def test_format_results_renders_table(populated, fake_client, collection):
    searcher = SemanticSearcher(client=fake_client, collection=collection)
    output = _format_results(searcher.search("Python", top_k=2))
    lines = output.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("1. [")
    assert lines[1].startswith("2. [")


def test_format_results_empty():
    assert _format_results([]) == "No results."


def test_cli_main_runs_end_to_end(populated, fake_client, collection, monkeypatch, capsys):
    # Patch the real-client builder so the CLI uses our fakes.
    from src import search as search_module

    def _fake_builder(collection_name, persist_directory, model):
        return SemanticSearcher(client=fake_client, collection=collection, model=model)

    monkeypatch.setattr(search_module, "_build_default_searcher", _fake_builder)

    exit_code = main(["Python web framework", "--top-k", "3"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    lines = [ln for ln in captured.strip().splitlines() if ln]
    assert len(lines) == 3
    assert lines[0].startswith("1. [")


def test_cli_json_output(populated, fake_client, collection, monkeypatch, capsys):
    import json

    from src import search as search_module

    def _fake_builder(collection_name, persist_directory, model):
        return SemanticSearcher(client=fake_client, collection=collection, model=model)

    monkeypatch.setattr(search_module, "_build_default_searcher", _fake_builder)

    exit_code = main(["ChromaDB vectors", "--top-k", "2", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 2
    assert {"id", "document", "score", "metadata"} <= set(payload[0].keys())
