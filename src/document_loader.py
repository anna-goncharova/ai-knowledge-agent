"""Recursively load documents from a directory.

Supported formats:
- ``.md`` / ``.txt`` / ``.rst`` — plain text (UTF-8)
- ``.pdf`` — parsed with PyPDF2
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf_file(path: Path) -> str:
    """Extract text from a PDF using PyPDF2.

    Non-extractable pages (images, scans) contribute an empty string so
    the returned value is always a ``str``.
    """
    from PyPDF2 import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover - defensive: malformed PDFs
            text = ""
        pages.append(text)
    return "\n".join(pages)


#: Mapping of lowercase file extension -> loader callable.
LOADERS: dict[str, Callable[[Path], str]] = {
    ".md": _load_text_file,
    ".txt": _load_text_file,
    ".rst": _load_text_file,
    ".pdf": _load_pdf_file,
}

SUPPORTED_EXTENSIONS = frozenset(LOADERS.keys())


@dataclass(frozen=True)
class Document:
    path: Path
    content: str


def load_document(path: str | Path) -> Document:
    """Load a single file by dispatching on its extension."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Path does not exist: {file_path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is not a file: {file_path}")
    ext = file_path.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return Document(path=file_path, content=loader(file_path))


def load_documents(root: str | Path) -> list[Document]:
    """Recursively load every supported file under ``root``.

    Returns a list of :class:`Document` objects sorted by path for
    deterministic ordering. Raises ``FileNotFoundError`` if ``root``
    does not exist and ``NotADirectoryError`` if it is not a directory.
    Unsupported file types are silently skipped.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    documents: list[Document] = []
    for file_path in sorted(root_path.rglob("*")):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        loader = LOADERS.get(ext)
        if loader is None:
            continue
        documents.append(Document(path=file_path, content=loader(file_path)))
    return documents
