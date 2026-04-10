"""Document loading built on top of LangChain loaders.

Wraps ``langchain_community`` loaders so the rest of the codebase keeps
using a tiny local :class:`Document` dataclass (``path``, ``content``)
and does not leak LangChain types beyond this module.

Supported formats:
- ``.md`` / ``.txt`` / ``.rst`` — loaded with ``TextLoader``
- ``.pdf`` — loaded with ``PyPDFLoader`` (backed by ``pypdf``)

Directory ingestion uses ``DirectoryLoader`` with one call per format
(each with its own ``loader_cls``), which keeps the dispatch explicit
and matches LangChain's recommended pattern for mixed-format folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document as LCDocument

# Mapping of lowercase extension -> LangChain BaseLoader class.
LOADER_CLASSES: dict[str, type] = {
    ".md": TextLoader,
    ".txt": TextLoader,
    ".rst": TextLoader,
    ".pdf": PyPDFLoader,
}

SUPPORTED_EXTENSIONS = frozenset(LOADER_CLASSES.keys())


@dataclass(frozen=True)
class Document:
    """Lightweight local document used by the rest of the app."""

    path: Path
    content: str


def _merge_lc_docs(lc_docs: list[LCDocument]) -> str:
    """Join LangChain ``Document`` chunks back into a single string.

    ``PyPDFLoader`` emits one :class:`LCDocument` per page; ``TextLoader``
    emits exactly one. We collapse both into a single ``content`` string
    so the downstream chunker/embedder see whole files.
    """
    return "\n".join(d.page_content for d in lc_docs)


def _load_single(path: Path) -> str:
    ext = path.suffix.lower()
    loader_cls = LOADER_CLASSES.get(ext)
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    # TextLoader defaults to autodetect encoding off; force utf-8.
    if loader_cls is TextLoader:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        loader = loader_cls(str(path))
    return _merge_lc_docs(loader.load())


def load_document(path: str | Path) -> Document:
    """Load a single file via the appropriate LangChain loader."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Path does not exist: {file_path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is not a file: {file_path}")
    return Document(path=file_path, content=_load_single(file_path))


def load_documents(root: str | Path) -> list[Document]:
    """Recursively load every supported file under ``root``.

    Uses one :class:`DirectoryLoader` per format (filtered via a glob
    pattern and wired to the format-specific ``loader_cls``). Unsupported
    files are silently skipped. Results are merged and sorted by path.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    documents: dict[Path, str] = {}
    for ext, loader_cls in LOADER_CLASSES.items():
        loader_kwargs: dict = {}
        if loader_cls is TextLoader:
            loader_kwargs["encoding"] = "utf-8"

        directory_loader = DirectoryLoader(
            str(root_path),
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            show_progress=False,
            use_multithreading=False,
        )
        lc_docs = directory_loader.load()

        # ``DirectoryLoader`` emits LC documents with the file path in
        # ``metadata["source"]``. Group per source path because PDF pages
        # arrive as multiple LC docs that share the same source.
        grouped: dict[Path, list[LCDocument]] = {}
        for lc in lc_docs:
            source = lc.metadata.get("source")
            if not source:
                continue
            src_path = Path(source)
            grouped.setdefault(src_path, []).append(lc)

        for src_path, lc_list in grouped.items():
            documents[src_path] = _merge_lc_docs(lc_list)

    return [
        Document(path=p, content=documents[p]) for p in sorted(documents.keys())
    ]


# ---------- Backwards-compatible aliases ----------
# Kept so pre-refactor imports (`LOADERS`, `_load_pdf_file`) don't break
# and so tests can patch at the old names too.

LOADERS: dict[str, Callable[[Path], str]] = {
    ext: (lambda p, _cls=cls: _load_single(p)) for ext, cls in LOADER_CLASSES.items()
}
