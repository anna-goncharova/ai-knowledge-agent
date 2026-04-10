from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from src import document_loader
from src.document_loader import (
    LOADERS,
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


def test_rst_is_registered_in_loaders():
    assert ".rst" in SUPPORTED_EXTENSIONS
    assert ".rst" in LOADERS


# ---------- .pdf loader (PyPDF2) ----------


@dataclass
class _FakePage:
    _text: str

    def extract_text(self) -> str:
        return self._text


class _FakePdfReader:
    """Stands in for ``PyPDF2.PdfReader`` in tests."""

    def __init__(self, source, pages_text: list[str]):
        self.source = source
        self.pages = [_FakePage(t) for t in pages_text]


def _install_fake_reader(monkeypatch, pages_text: list[str]) -> list:
    """Patch ``PyPDF2.PdfReader`` to return a FakePdfReader.

    Returns a list that records the source paths PdfReader was called with.
    """
    calls: list = []

    def _factory(source):
        calls.append(source)
        return _FakePdfReader(source, pages_text)

    import PyPDF2

    monkeypatch.setattr(PyPDF2, "PdfReader", _factory)
    return calls


def test_pdf_loader_extracts_text_from_all_pages(tmp_path: Path, monkeypatch):
    calls = _install_fake_reader(
        monkeypatch,
        pages_text=["Page one text.", "Page two text.", "Page three."],
    )
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf bytes")

    doc = load_document(pdf_path)

    assert doc.path == pdf_path
    assert doc.content == "Page one text.\nPage two text.\nPage three."
    # PdfReader was called once with the pdf path (as str).
    assert calls == [str(pdf_path)]


def test_pdf_loader_handles_none_and_errors(tmp_path: Path, monkeypatch):
    class _BrokenPage:
        def extract_text(self):
            raise RuntimeError("corrupt page")

    class _NonePage:
        def extract_text(self):
            return None

    class _MixedReader:
        def __init__(self, source):
            self.pages = [_FakePage("ok page"), _NonePage(), _BrokenPage()]

    import PyPDF2

    monkeypatch.setattr(PyPDF2, "PdfReader", _MixedReader)

    pdf_path = tmp_path / "mixed.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    doc = load_document(pdf_path)

    # None -> "", broken -> "". Empty lines are preserved between pages.
    assert doc.content == "ok page\n\n"


def test_pdf_loader_via_recursive_load(tmp_path: Path, monkeypatch):
    _install_fake_reader(monkeypatch, pages_text=["hello from pdf"])
    (tmp_path / "a.md").write_text("markdown content", encoding="utf-8")
    (tmp_path / "b.rst").write_text("rst content", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "skip.docx").write_bytes(b"not handled")

    docs = load_documents(tmp_path)
    by_name = {d.path.name: d.content for d in docs}
    assert set(by_name) == {"a.md", "b.rst", "c.pdf"}
    assert by_name["c.pdf"] == "hello from pdf"


def test_pdf_is_registered_in_loaders():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert LOADERS[".pdf"] is document_loader._load_pdf_file


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
