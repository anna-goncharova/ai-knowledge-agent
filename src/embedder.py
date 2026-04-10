"""Embed text chunks via OpenAI and store them in ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import chromadb
from chromadb.api.models.Collection import Collection

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_COLLECTION_NAME = "ai_knowledge_agent"


class EmbeddingClient(Protocol):
    """Minimal protocol for an OpenAI-compatible embeddings client.

    Allows injecting a fake in tests. ``openai.OpenAI`` satisfies this
    because it exposes ``client.embeddings.create(...)``.
    """

    embeddings: "EmbeddingsResource"


class EmbeddingsResource(Protocol):
    def create(self, *, model: str, input: Sequence[str]): ...  # noqa: D401


@dataclass
class Embedder:
    """Embeds chunks and upserts them into a ChromaDB collection."""

    client: EmbeddingClient
    collection: Collection
    model: str = DEFAULT_EMBEDDING_MODEL

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]

    def add_chunks(
        self,
        chunks: Sequence[str],
        ids: Sequence[str] | None = None,
        metadatas: Sequence[dict] | None = None,
    ) -> list[str]:
        """Embed ``chunks`` and store them in the collection.

        Returns the list of ids that were inserted.
        """
        if not chunks:
            return []
        if ids is None:
            ids = [f"chunk-{i}" for i in range(len(chunks))]
        if len(ids) != len(chunks):
            raise ValueError("ids and chunks must have the same length")
        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError("metadatas and chunks must have the same length")

        embeddings = self.embed(chunks)
        self.collection.add(
            ids=list(ids),
            embeddings=embeddings,
            documents=list(chunks),
            metadatas=list(metadatas) if metadatas is not None else None,
        )
        return list(ids)


def build_openai_embedder(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
) -> Embedder:
    """Build an :class:`Embedder` using the real OpenAI client and a
    persistent or ephemeral ChromaDB collection.

    If ``persist_directory`` is ``None``, an in-memory client is used.
    """
    from openai import OpenAI

    openai_client = OpenAI(api_key=api_key) if api_key else OpenAI()
    if persist_directory:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
    else:
        chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(collection_name)
    return Embedder(client=openai_client, collection=collection, model=model)
