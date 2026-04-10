"""Semantic search: query -> OpenAI embedding -> ChromaDB top-k results."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from typing import Sequence

from chromadb.api.models.Collection import Collection

from src.embedder import DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL

DEFAULT_TOP_K = 5


@dataclass
class SearchResult:
    id: str
    document: str
    score: float
    metadata: dict | None = None


@dataclass
class SemanticSearcher:
    """Embed a query and retrieve the top-k nearest chunks from ChromaDB."""

    client: object  # OpenAI-compatible client with .embeddings.create(...)
    collection: Collection
    model: str = DEFAULT_EMBEDDING_MODEL

    def _embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=[query])
        return response.data[0].embedding

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_embedding = self._embed_query(query)
        raw = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        ids: Sequence[str] = raw["ids"][0]
        docs: Sequence[str] = raw["documents"][0]
        distances: Sequence[float] = raw["distances"][0]
        metadatas: Sequence[dict] | None = (
            raw["metadatas"][0] if raw.get("metadatas") else None
        )

        results: list[SearchResult] = []
        for i, doc_id in enumerate(ids):
            # Convert distance to a similarity-like score in [0, 1] where
            # 1.0 means identical. ChromaDB distances are non-negative.
            distance = float(distances[i])
            score = 1.0 / (1.0 + distance)
            results.append(
                SearchResult(
                    id=doc_id,
                    document=docs[i],
                    score=score,
                    metadata=metadatas[i] if metadatas else None,
                )
            )
        return results


def _build_default_searcher(
    collection_name: str,
    persist_directory: str | None,
    model: str,
) -> SemanticSearcher:
    """Wire up a real OpenAI client and a ChromaDB collection."""
    import chromadb
    from openai import OpenAI

    openai_client = OpenAI()
    if persist_directory:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
    else:
        chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(collection_name)
    return SemanticSearcher(client=openai_client, collection=collection, model=model)


def _format_results(results: list[SearchResult]) -> str:
    if not results:
        return "No results."
    lines = []
    for rank, r in enumerate(results, start=1):
        preview = r.document.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:197] + "..."
        lines.append(f"{rank}. [{r.score:.4f}] id={r.id}  {preview}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-knowledge-search",
        description="Semantic search over an indexed ChromaDB collection.",
    )
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION_NAME})",
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Path to a persistent ChromaDB directory (default: in-memory)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"OpenAI embedding model (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of a table",
    )
    args = parser.parse_args(argv)

    searcher = _build_default_searcher(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        model=args.model,
    )
    results = searcher.search(args.query, top_k=args.top_k)

    if args.json:
        import json

        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
    else:
        print(_format_results(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
