"""Tests for src.document_loader (LangChain-backed)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document as LCDocument

from src import document_loader
from src.document_loader import (
    LOADER_CLASSES,
    SUPPORTED_EXTENSIONS,
    Document,
    load_document,
    load_documents,
)


# ---------- md / txt / rst (text loaders) ----------


def test_loads_md_txt_and_rst_recursively(tmp_path: Path):
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "c.rst").write_text("RST title\n=========\n", encoding="utf-8")
    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    (nested / "d.md").write_text("nested md", encoding="utf-8")
    (nested / "e.rst").write_text("nested rst", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert len(docs) == 5
    assert all(isinstance(d, Document) for d in docs)
    contents = {d.path.name: d.content for d in docs}
    assert contents == {
        "a.md": "# A",
        "b.txt": "hello",
        "c.rst": "RST title\n=========\n",
        "d.md": "nested md",
        "e.rst": "nested rst",
    }


def test_ignores_unsupported_extensions(tmp_path: Path):
    (tmp_path / "keep.md").write_text("keep", encoding="utf-8")
    (tmp_path / "keep.rst").write_text("rst", encoding="utf-8")
    (tmp_path / "skip.py").write_text("print('x')", encoding="utf-8")
    (tmp_path / "skip.html").write_text("<p>x</p>", encoding="utf-8")
    (tmp_path / "skip.json").write_text("{}", encoding="utf-8")

    docs = load_documents(tmp_path)
    names = sorted(d.path.name for d in docs)
    assert names == ["keep.md", "keep.rst"]


def test_raises_for_missing_or_invalid_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_documents(missing)

    file_path = tmp_path / "file.md"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        load_documents(file_path)


# ---------- .rst loader ----------


def test_rst_loader_returns_raw_content(tmp_path: Path):
    rst_body = (
        "Project Title\n"
        "=============\n\n"
        "Section\n"
        "-------\n\n"
        "- bullet one\n"
        "- bullet two\n"
    )
    path = tmp_path / "guide.rst"
    path.write_text(rst_body, encoding="utf-8")

    doc = load_document(path)
    assert doc.path == path
    assert doc.content == rst_body


def test_rst_is_registered_in_loader_classes():
    from langchain_community.document_loaders import TextLoader

    assert ".rst" in SUPPORTED_EXTENSIONS
    assert LOADER_CLASSES[".rst"] is TextLoader


# ---------- .pdf loader (PyPDFLoader) ----------


class _FakePyPDFLoader:
    """Stands in for ``PyPDFLoader`` — returns one LCDocument per page."""

    pages_by_path: dict[str, list[str]] = {}

    def __init__(self, file_path: str, **kwargs):
        self.file_path = file_path

    def load(self) -> list[LCDocument]:
        pages = self.pages_by_path.get(
            self.file_path, ["default page 1", "default page 2"]
        )
        return [
            LCDocument(
                page_content=text,
                metadata={"source": self.file_path, "page": i},
            )
            for i, text in enumerate(pages)
        ]

    def lazy_load(self):
        # DirectoryLoader uses lazy_load() under the hood.
        yield from self.load()


@pytest.fixture
def fake_pypdf_loader(monkeypatch):
    """Replace PyPDFLoader in both LOADER_CLASSES and the module import path."""
    _FakePyPDFLoader.pages_by_path = {}
    monkeypatch.setitem(LOADER_CLASSES, ".pdf", _FakePyPDFLoader)
    monkeypatch.setattr(document_loader, "PyPDFLoader", _FakePyPDFLoader)
    return _FakePyPDFLoader


def test_pdf_loader_extracts_text_from_all_pages(
    tmp_path: Path, fake_pypdf_loader
):
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf bytes")
    fake_pypdf_loader.pages_by_path[str(pdf_path)] = [
        "Page one text.",
        "Page two text.",
        "Page three.",
    ]

    doc = load_document(pdf_path)

    assert doc.path == pdf_path
    assert doc.content == "Page one text.\nPage two text.\nPage three."


def test_pdf_loader_single_page(tmp_path: Path, fake_pypdf_loader):
    pdf_path = tmp_path / "single.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    fake_pypdf_loader.pages_by_path[str(pdf_path)] = ["only page"]

    doc = load_document(pdf_path)
    assert doc.content == "only page"


def test_pdf_loader_via_recursive_load(tmp_path: Path, fake_pypdf_loader):
    (tmp_path / "a.md").write_text("markdown content", encoding="utf-8")
    (tmp_path / "b.rst").write_text("rst content", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "skip.docx").write_bytes(b"not handled")

    pdf_str = str(tmp_path / "c.pdf")
    fake_pypdf_loader.pages_by_path[pdf_str] = ["hello", "from pdf"]

    docs = load_documents(tmp_path)
    by_name = {d.path.name: d.content for d in docs}
    assert set(by_name) == {"a.md", "b.rst", "c.pdf"}
    assert by_name["c.pdf"] == "hello\nfrom pdf"
    assert by_name["a.md"] == "markdown content"
    assert by_name["b.rst"] == "rst content"


def test_pdf_is_registered_in_loader_classes():
    from langchain_community.document_loaders import PyPDFLoader

    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert LOADER_CLASSES[".pdf"] is PyPDFLoader


# ---------- load_document dispatch ----------


def test_load_document_rejects_unknown_extension(tmp_path: Path):
    path = tmp_path / "file.xyz"
    path.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(path)


def test_load_document_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "ghost.md")


def test_load_document_rejects_directory(tmp_path: Path):
    with pytest.raises(IsADirectoryError):
        load_document(tmp_path)
