"""Retrieval-augmented QA: semantic search -> LLM with context -> answer."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Sequence

from src.embedder import DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL
from src.search import DEFAULT_TOP_K, SearchResult, SemanticSearcher

DEFAULT_CHAT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer the user's question using ONLY the "
    "provided context chunks. If the answer is not in the context, say you don't "
    "know. Cite sources by their [n] index when useful. Be concise."
)


@dataclass
class QAResponse:
    question: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)


def _build_context(results: Sequence[SearchResult]) -> str:
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] (id={r.id}, score={r.score:.4f})\n{r.document}")
    return "\n\n".join(lines)


def _build_user_prompt(question: str, context: str) -> str:
    if not context:
        return f"Question: {question}\n\n(No context was retrieved.)"
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


@dataclass
class QAChain:
    """Retrieve top-k chunks for a question and ask an LLM to answer."""

    searcher: SemanticSearcher
    client: object  # OpenAI-compatible client with .chat.completions.create(...)
    chat_model: str = DEFAULT_CHAT_MODEL
    top_k: int = DEFAULT_TOP_K
    system_prompt: str = SYSTEM_PROMPT

    def ask(self, question: str) -> QAResponse:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        sources = self.searcher.search(question, top_k=self.top_k)
        context = _build_context(sources)
        user_prompt = _build_user_prompt(question, context)

        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = completion.choices[0].message.content
        return QAResponse(question=question, answer=answer, sources=sources)


def _build_default_chain(
    collection_name: str,
    persist_directory: str | None,
    embedding_model: str,
    chat_model: str,
    top_k: int,
) -> QAChain:
    """Wire up real OpenAI client + ChromaDB for production use."""
    import chromadb
    from openai import OpenAI

    openai_client = OpenAI()
    if persist_directory:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
    else:
        chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(collection_name)
    searcher = SemanticSearcher(
        client=openai_client, collection=collection, model=embedding_model
    )
    return QAChain(
        searcher=searcher,
        client=openai_client,
        chat_model=chat_model,
        top_k=top_k,
    )


def _format_response(response: QAResponse) -> str:
    lines = [f"Answer: {response.answer}"]
    if response.sources:
        lines.append("\nSources:")
        for i, s in enumerate(response.sources, start=1):
            preview = s.document.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            lines.append(f"  [{i}] ({s.score:.4f}) {s.id}  {preview}")
    return "\n".join(lines)


def _chat_loop(chain: QAChain, input_fn=input, output_fn=print) -> int:
    """Interactive REPL over a :class:`QAChain`.

    ``input_fn`` and ``output_fn`` are parameterised for testability.
    """
    output_fn("AI Knowledge Agent — chat mode. Type 'exit' or 'quit' to leave.")
    while True:
        try:
            question = input_fn("you> ")
        except EOFError:
            output_fn("")
            return 0
        if question is None:
            return 0
        question = question.strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            output_fn("bye")
            return 0
        try:
            response = chain.ask(question)
        except Exception as exc:  # pragma: no cover - defensive
            output_fn(f"error: {exc}")
            continue
        output_fn(_format_response(response))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-knowledge-qa",
        description="RAG QA chat over an indexed ChromaDB collection.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="One-shot question; omit to start interactive chat.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    args = parser.parse_args(argv)

    chain = _build_default_chain(
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        top_k=args.top_k,
    )

    if args.question:
        response = chain.ask(args.question)
        print(_format_response(response))
        return 0

    return _chat_loop(chain)


if __name__ == "__main__":
    sys.exit(main())
