"""Tests for the FastAPI wrapper using httpx.AsyncClient + ASGITransport.

Covers:
- /ask, /upload (file), /search endpoints
- API-key auth via the X-API-Key header
- Pydantic validation edge cases
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

import chromadb
import httpx
import pytest
import pytest_asyncio

from src.embedder import Embedder
from src.main import AppContext, app, set_api_key
from src.qa_chain import QAChain
from src.search import SemanticSearcher

API_KEY = "test-secret-key"
EMBED_DIM = 16


# ---------- Fake OpenAI client (no network calls) ----------


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
    def __init__(self, canned_answer: str = "RAG answer from stub."):
        self.embeddings = _FakeEmbeddingsResource()
        self.chat = _FakeChat(_FakeChatCompletionsResource(canned_answer))


# ---------- Fixtures: configured app + httpx.AsyncClient over ASGITransport ----------


@pytest.fixture
def configured_app(request):
    """Attach a fresh AppContext + API key to the module-level ``app``."""
    fake_client = _FakeOpenAIClient()
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
    set_api_key(app, API_KEY)
    try:
        yield app
    finally:
        app.state.ctx = None
        set_api_key(app, None)


@pytest_asyncio.fixture
async def http(configured_app) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=configured_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"X-API-Key": API_KEY},
    ) as client:
        yield client


@pytest_asyncio.fixture
async def http_no_auth(configured_app) -> AsyncIterator[httpx.AsyncClient]:
    """httpx.AsyncClient without the API key header preset."""
    transport = httpx.ASGITransport(app=configured_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client


# ---------- Public endpoints (no auth) ----------


async def test_root_and_health_do_not_require_auth(http_no_auth):
    assert (await http_no_auth.get("/")).status_code == 200
    assert (await http_no_auth.get("/health")).json() == {"status": "ok"}


# ---------- Auth ----------


async def test_upload_requires_api_key(http_no_auth):
    files = {"file": ("notes.md", b"hello world", "text/markdown")}
    r = await http_no_auth.post("/upload", files=files)
    assert r.status_code == 401
    assert "Missing" in r.json()["detail"]


async def test_upload_rejects_wrong_api_key(http_no_auth):
    files = {"file": ("notes.md", b"hello world", "text/markdown")}
    r = await http_no_auth.post(
        "/upload", files=files, headers={"X-API-Key": "nope"}
    )
    assert r.status_code == 403


async def test_ask_and_search_require_api_key(http_no_auth):
    assert (await http_no_auth.post("/ask", json={"question": "hi"})).status_code == 401
    assert (await http_no_auth.get("/search", params={"query": "hi"})).status_code == 401


async def test_auth_disabled_when_api_key_unset(configured_app):
    # Disable auth and verify endpoints become publicly accessible.
    set_api_key(configured_app, None)
    transport = httpx.ASGITransport(app=configured_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as c:
        r = await c.post(
            "/upload",
            files={"file": ("x.md", b"hello world", "text/markdown")},
        )
        assert r.status_code == 200


# ---------- /upload ----------


async def test_upload_file_chunks_and_stores(http):
    body = (" ".join(f"word{i:04d}" for i in range(300))).encode("utf-8")
    files = {"file": ("notes.md", body, "text/markdown")}
    r = await http.post("/upload", files=files)
    assert r.status_code == 200

    payload = r.json()
    assert payload["source"] == "notes.md"
    assert payload["filename"] == "notes.md"
    assert payload["chunks_added"] >= 1
    assert len(payload["ids"]) == payload["chunks_added"]
    assert all(i.startswith("notes.md::chunk-") for i in payload["ids"])


async def test_upload_txt_with_form_overrides(http):
    files = {"file": ("big.txt", b"a" * 1000, "text/plain")}
    data = {"chunk_size": "100", "chunk_overlap": "20", "source": "custom-name"}
    r = await http.post("/upload", files=files, data=data)
    assert r.status_code == 200
    payload = r.json()
    assert payload["source"] == "custom-name"
    assert payload["filename"] == "big.txt"
    assert payload["chunks_added"] > 1
    assert all(i.startswith("custom-name::chunk-") for i in payload["ids"])


async def test_upload_rejects_unsupported_extension(http):
    files = {"file": ("picture.png", b"\x89PNG\r\n\x1a\n", "image/png")}
    r = await http.post("/upload", files=files)
    assert r.status_code == 415


async def test_upload_rejects_non_utf8(http):
    files = {"file": ("bad.md", b"\xff\xfe\xfa", "text/markdown")}
    r = await http.post("/upload", files=files)
    assert r.status_code == 400
    assert "UTF-8" in r.json()["detail"]


async def test_upload_rejects_empty_file(http):
    files = {"file": ("empty.md", b"   \n", "text/markdown")}
    r = await http.post("/upload", files=files)
    assert r.status_code == 400


async def test_upload_requires_file_field(http):
    r = await http.post("/upload")
    assert r.status_code == 422


# ---------- /search ----------


async def test_search_after_upload(http):
    content = "\n\n".join(
        [
            "FastAPI is a modern Python web framework.",
            "ChromaDB stores vector embeddings.",
            "OpenAI provides embedding models.",
            "RAG combines retrieval and generation.",
            "LangChain offers text splitters and chains.",
        ]
    ).encode("utf-8")
    await http.post(
        "/upload",
        files={"file": ("kb.md", content, "text/markdown")},
        data={"chunk_size": "50", "chunk_overlap": "10"},
    )

    r = await http.get("/search", params={"query": "vector database", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "vector database"
    assert len(body["results"]) == 3
    for result in body["results"]:
        assert {"id", "document", "score", "metadata"} <= set(result.keys())
        assert 0.0 < result["score"] <= 1.0
    scores = [item["score"] for item in body["results"]]
    assert scores == sorted(scores, reverse=True)


async def test_search_validates_params(http):
    assert (await http.get("/search")).status_code == 422
    assert (
        await http.get("/search", params={"query": "x", "top_k": 0})
    ).status_code == 422


# ---------- /ask ----------


async def test_ask_returns_answer_and_sources(http):
    content = "\n\n".join(f"doc number {i}: lorem ipsum" for i in range(10)).encode(
        "utf-8"
    )
    await http.post(
        "/upload",
        files={"file": ("seed.md", content, "text/markdown")},
        data={"chunk_size": "30", "chunk_overlap": "5"},
    )

    r = await http.post(
        "/ask",
        json={"question": "What is the meaning of life?", "top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "What is the meaning of life?"
    assert body["answer"] == "RAG answer from stub."
    assert len(body["sources"]) == 3


async def test_ask_rejects_empty_question(http):
    r = await http.post("/ask", json={"question": ""})
    assert r.status_code == 422


# ---------- Missing context ----------


async def test_endpoints_return_503_without_context():
    app.state.ctx = None
    set_api_key(app, None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as c:
        assert (await c.post("/ask", json={"question": "hi"})).status_code == 503
        r = await c.post(
            "/upload",
            files={"file": ("x.md", b"hi", "text/markdown")},
        )
        assert r.status_code == 503
        assert (await c.get("/search", params={"query": "x"})).status_code == 503
