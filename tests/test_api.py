"""Integration tests for the FastAPI wrapper (/ask, /upload, /search)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import chromadb
import pytest
from fastapi.testclient import TestClient

from src.embedder import Embedder
from src.main import AppContext, app
from src.qa_chain import QAChain
from src.search import SemanticSearcher

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
    def create(self, *, model: str, input: Sequence[str]):
        return _FakeEmbeddingsResponse(
            data=[_FakeEmbeddingItem(_deterministic_embedding(t)) for t in input]
        )


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeChatCompletion:
    choices: list[_FakeChoice]


class _FakeChatCompletionsResource:
    def __init__(self, canned_answer: str):
        self.canned_answer = canned_answer

    def create(self, *, model: str, messages: list[dict]):
        return _FakeChatCompletion(
            choices=[_FakeChoice(message=_FakeMessage(content=self.canned_answer))]
        )


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, canned_answer: str = "stubbed answer"):
        self.embeddings = _FakeEmbeddingsResource()
        self.chat = _FakeChat(_FakeChatCompletionsResource(canned_answer))


@pytest.fixture
def client(request):
    fake_client = _FakeOpenAIClient(canned_answer="RAG answer from stub.")
    chroma_client = chromadb.EphemeralClient()
    name = f"test_api_{request.node.name}"
    try:
        chroma_client.delete_collection(name)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(name)

    embedder = Embedder(client=fake_client, collection=collection)
    searcher = SemanticSearcher(client=fake_client, collection=collection)
    qa_chain = QAChain(searcher=searcher, client=fake_client)
    app.state.ctx = AppContext(embedder=embedder, searcher=searcher, qa_chain=qa_chain)

    with TestClient(app) as c:
        yield c

    # Clean up the context so tests don't bleed state.
    app.state.ctx = None


def test_root_and_health(client):
    assert client.get("/").json() == {"message": "Hello, World!"}
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_chunks_and_stores(client):
    content = " ".join([f"word{i:04d}" for i in range(300)])
    response = client.post(
        "/upload", json={"source": "notes.md", "content": content}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "notes.md"
    assert body["chunks_added"] >= 1
    assert len(body["ids"]) == body["chunks_added"]
    assert all(i.startswith("notes.md::chunk-") for i in body["ids"])


def test_upload_custom_chunk_params(client):
    content = "a" * 1000
    response = client.post(
        "/upload",
        json={
            "source": "big.txt",
            "content": content,
            "chunk_size": 100,
            "chunk_overlap": 20,
        },
    )
    assert response.status_code == 200
    assert response.json()["chunks_added"] > 1


def test_upload_rejects_empty_content(client):
    response = client.post("/upload", json={"source": "x", "content": ""})
    assert response.status_code == 422  # pydantic validation


def test_search_after_upload_returns_results_with_scores(client):
    content = "\n\n".join(
        [
            "FastAPI is a modern Python web framework.",
            "ChromaDB stores vector embeddings.",
            "OpenAI provides embedding models like text-embedding-3-small.",
            "RAG combines retrieval and generation for grounded answers.",
            "LangChain offers text splitters and chains.",
        ]
    )
    # Force small chunks so we get multiple rows in the collection.
    upload = client.post(
        "/upload",
        json={
            "source": "kb.md",
            "content": content,
            "chunk_size": 50,
            "chunk_overlap": 10,
        },
    )
    assert upload.json()["chunks_added"] >= 3

    response = client.get("/search", params={"query": "vector database", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "vector database"
    assert len(body["results"]) == 3
    for result in body["results"]:
        assert set(result.keys()) >= {"id", "document", "score", "metadata"}
        assert 0.0 < result["score"] <= 1.0
    # Results are ordered by descending score.
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_validates_params(client):
    # Missing required query.
    assert client.get("/search").status_code == 422
    # Invalid top_k.
    assert (
        client.get("/search", params={"query": "x", "top_k": 0}).status_code == 422
    )


def test_ask_returns_answer_and_sources(client):
    content = "\n\n".join([f"doc number {i}: lorem ipsum" for i in range(10)])
    client.post(
        "/upload",
        json={
            "source": "seed.md",
            "content": content,
            "chunk_size": 30,
            "chunk_overlap": 5,
        },
    )

    response = client.post(
        "/ask", json={"question": "What is the meaning of life?", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "What is the meaning of life?"
    assert body["answer"] == "RAG answer from stub."
    assert len(body["sources"]) == 3


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_endpoints_return_503_without_context():
    # Fresh client with no context attached.
    app.state.ctx = None
    with TestClient(app) as c:
        assert c.post("/ask", json={"question": "hi"}).status_code == 503
        assert (
            c.post("/upload", json={"source": "x", "content": "y"}).status_code == 503
        )
        assert c.get("/search", params={"query": "x"}).status_code == 503
