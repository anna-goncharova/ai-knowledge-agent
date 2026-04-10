"""Tests for src.qa_chain with fake OpenAI client and in-memory ChromaDB."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import chromadb
import pytest

from src.embedder import Embedder
from src.qa_chain import (
    DEFAULT_CHAT_MODEL,
    QAChain,
    _build_context,
    _chat_loop,
    _format_response,
    main,
)
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
    def __init__(self, canned_answer: str = "42"):
        self.canned_answer = canned_answer
        self.calls: list[dict] = []

    def create(self, *, model: str, messages: list[dict]):
        self.calls.append({"model": model, "messages": messages})
        return _FakeChatCompletion(
            choices=[_FakeChoice(message=_FakeMessage(content=self.canned_answer))]
        )


class _FakeChatResource:
    def __init__(self, completions: _FakeChatCompletionsResource):
        self.completions = completions


class _FakeOpenAIClient:
    def __init__(self, canned_answer: str = "42"):
        self.embeddings = _FakeEmbeddingsResource()
        self.chat = _FakeChatResource(_FakeChatCompletionsResource(canned_answer))


@pytest.fixture
def collection(request):
    client = chromadb.EphemeralClient()
    name = f"test_qa_{request.node.name}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name)


@pytest.fixture
def fake_client():
    return _FakeOpenAIClient(canned_answer="FastAPI is a Python web framework.")


@pytest.fixture
def populated_chain(fake_client, collection):
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
    Embedder(client=fake_client, collection=collection).add_chunks(docs)
    searcher = SemanticSearcher(client=fake_client, collection=collection)
    chain = QAChain(searcher=searcher, client=fake_client)
    return chain, docs


def test_ask_returns_answer_and_sources(populated_chain, fake_client):
    chain, _ = populated_chain
    response = chain.ask("What is FastAPI?")

    assert response.question == "What is FastAPI?"
    assert response.answer == "FastAPI is a Python web framework."
    assert len(response.sources) == 5  # default top_k
    # LLM was called once with the configured chat model.
    calls = fake_client.chat.completions.calls
    assert len(calls) == 1
    assert calls[0]["model"] == DEFAULT_CHAT_MODEL
    messages = calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    # User prompt embeds both the question and the retrieved context.
    assert "What is FastAPI?" in messages[1]["content"]
    assert "Context:" in messages[1]["content"]
    # Context contains 5 numbered chunks from the populated collection.
    for i in range(1, 6):
        assert f"[{i}]" in messages[1]["content"]


def test_context_includes_top_k_chunks_with_scores(populated_chain):
    chain, _ = populated_chain
    results = chain.searcher.search("FastAPI", top_k=3)
    context = _build_context(results)
    assert "[1]" in context and "[2]" in context and "[3]" in context
    assert "score=" in context


def test_custom_top_k_limits_sources(fake_client, collection):
    docs = [f"document number {i}" for i in range(10)]
    Embedder(client=fake_client, collection=collection).add_chunks(docs)
    searcher = SemanticSearcher(client=fake_client, collection=collection)
    chain = QAChain(searcher=searcher, client=fake_client, top_k=2)

    response = chain.ask("anything")
    assert len(response.sources) == 2


def test_empty_question_raises(populated_chain):
    chain, _ = populated_chain
    with pytest.raises(ValueError):
        chain.ask("")
    with pytest.raises(ValueError):
        chain.ask("   ")


def test_format_response_renders_sources(populated_chain):
    chain, _ = populated_chain
    response = chain.ask("FastAPI?")
    text = _format_response(response)
    assert "Answer:" in text
    assert "Sources:" in text
    # All 5 default sources are listed.
    for i in range(1, 6):
        assert f"[{i}]" in text


def test_cli_one_shot(populated_chain, monkeypatch, capsys):
    chain, _ = populated_chain
    from src import qa_chain as qa_module

    def _fake_builder(**kwargs):
        return chain

    monkeypatch.setattr(qa_module, "_build_default_chain", _fake_builder)

    exit_code = main(["What is FastAPI?", "--top-k", "3"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Answer: FastAPI is a Python web framework." in out
    assert "Sources:" in out


def test_chat_loop_interactive(populated_chain):
    chain, _ = populated_chain

    inputs = iter(["", "What is FastAPI?", "exit"])
    outputs: list[str] = []

    def fake_input(prompt: str) -> str:
        return next(inputs)

    def fake_output(msg: str) -> None:
        outputs.append(msg)

    exit_code = _chat_loop(chain, input_fn=fake_input, output_fn=fake_output)
    assert exit_code == 0
    joined = "\n".join(outputs)
    assert "chat mode" in joined
    assert "FastAPI is a Python web framework." in joined
    assert "bye" in joined


def test_chat_loop_eof_exits_cleanly(populated_chain):
    chain, _ = populated_chain

    def fake_input(prompt: str) -> str:
        raise EOFError

    outputs: list[str] = []

    exit_code = _chat_loop(chain, input_fn=fake_input, output_fn=outputs.append)
    assert exit_code == 0
    assert any("chat mode" in o for o in outputs)
