"""Recursively load .md and .txt documents from a directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class Document:
    path: Path
    content: str


def load_documents(root: str | Path) -> list[Document]:
    """Recursively load all .md and .txt files under ``root``.

    Returns a list of :class:`Document` objects sorted by path for
    deterministic ordering. Raises ``FileNotFoundError`` if ``root``
    does not exist and ``NotADirectoryError`` if it is not a directory.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root_path}")

    documents: list[Document] = []
    for file_path in sorted(root_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            content = file_path.read_text(encoding="utf-8")
            documents.append(Document(path=file_path, content=content))
    return documents
