"""FastAPI wrapper exposing the ingestion, search, and QA pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.embedder import DEFAULT_COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL, Embedder
from src.qa_chain import DEFAULT_CHAT_MODEL, QAChain
from src.search import DEFAULT_TOP_K, SearchResult, SemanticSearcher
from src.text_chunker import chunk_text


# ---------- Application context (dependency-injected) ----------


@dataclass
class AppContext:
    """Shared pipeline objects used by the FastAPI endpoints."""

    embedder: Embedder
    searcher: SemanticSearcher
    qa_chain: QAChain


def get_context(request: Request) -> AppContext:
    ctx: AppContext | None = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise HTTPException(
            status_code=503,
            detail="AppContext not initialised. Call init_app_context() first.",
        )
    return ctx


def init_app_context(fastapi_app: FastAPI, context: AppContext) -> None:
    """Attach a pre-built :class:`AppContext` to the app (used in tests)."""
    fastapi_app.state.ctx = context


def build_default_context(
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chat_model: str = DEFAULT_CHAT_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> AppContext:
    """Build a production :class:`AppContext` with real OpenAI + ChromaDB."""
    import chromadb
    from openai import OpenAI

    openai_client = OpenAI()
    if persist_directory:
        chroma_client = chromadb.PersistentClient(path=persist_directory)
    else:
        chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(collection_name)

    embedder = Embedder(
        client=openai_client, collection=collection, model=embedding_model
    )
    searcher = SemanticSearcher(
        client=openai_client, collection=collection, model=embedding_model
    )
    qa_chain = QAChain(
        searcher=searcher,
        client=openai_client,
        chat_model=chat_model,
        top_k=top_k,
    )
    return AppContext(embedder=embedder, searcher=searcher, qa_chain=qa_chain)


# ---------- Pydantic models ----------


class SourceModel(BaseModel):
    id: str
    document: str
    score: float
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_result(cls, r: SearchResult) -> "SourceModel":
        return cls(**asdict(r))


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]


class UploadRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Logical source name, e.g. filename")
    content: str = Field(..., min_length=1)
    chunk_size: int | None = Field(default=None, ge=1)
    chunk_overlap: int | None = Field(default=None, ge=0)


class UploadResponse(BaseModel):
    source: str
    chunks_added: int
    ids: list[str]


class SearchResponseModel(BaseModel):
    query: str
    results: list[SourceModel]


# ---------- FastAPI app ----------


app = FastAPI(
    title="AI Knowledge Agent",
    description="AI-powered documentation audit tool",
    version="0.1.0",
)


@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, ctx: AppContext = Depends(get_context)) -> AskResponse:
    chain = ctx.qa_chain
    if req.top_k is not None and req.top_k != chain.top_k:
        chain = QAChain(
            searcher=chain.searcher,
            client=chain.client,
            chat_model=chain.chat_model,
            top_k=req.top_k,
            system_prompt=chain.system_prompt,
        )
    try:
        response = chain.ask(req.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AskResponse(
        question=response.question,
        answer=response.answer,
        sources=[SourceModel.from_result(s) for s in response.sources],
    )


@app.post("/upload", response_model=UploadResponse)
def upload(req: UploadRequest, ctx: AppContext = Depends(get_context)) -> UploadResponse:
    chunk_kwargs: dict[str, Any] = {}
    if req.chunk_size is not None:
        chunk_kwargs["chunk_size"] = req.chunk_size
    if req.chunk_overlap is not None:
        chunk_kwargs["chunk_overlap"] = req.chunk_overlap

    try:
        chunks = chunk_text(req.content, **chunk_kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from content")

    ids = [f"{req.source}::chunk-{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": req.source, "chunk_index": i} for i in range(len(chunks))
    ]
    ctx.embedder.add_chunks(chunks, ids=ids, metadatas=metadatas)
    return UploadResponse(source=req.source, chunks_added=len(chunks), ids=ids)


@app.get("/search", response_model=SearchResponseModel)
def search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=50),
    ctx: AppContext = Depends(get_context),
) -> SearchResponseModel:
    try:
        results = ctx.searcher.search(query, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SearchResponseModel(
        query=query,
        results=[SourceModel.from_result(r) for r in results],
    )
