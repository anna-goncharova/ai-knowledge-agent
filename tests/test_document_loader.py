from pathlib import Path

import pytest

from src.document_loader import Document, load_documents


def test_loads_md_and_txt_recursively(tmp_path: Path):
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello", encoding="utf-8")
    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    (nested / "c.md").write_text("nested md", encoding="utf-8")
    (nested / "d.txt").write_text("nested txt", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert len(docs) == 4
    assert all(isinstance(d, Document) for d in docs)
    contents = {d.path.name: d.content for d in docs}
    assert contents == {
        "a.md": "# A",
        "b.txt": "hello",
        "c.md": "nested md",
        "d.txt": "nested txt",
    }


def test_ignores_unsupported_extensions(tmp_path: Path):
    (tmp_path / "keep.md").write_text("keep", encoding="utf-8")
    (tmp_path / "skip.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "skip.py").write_text("print('x')", encoding="utf-8")
    (tmp_path / "skip.rst").write_text("rst", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert [d.path.name for d in docs] == ["keep.md"]


def test_raises_for_missing_or_invalid_path(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_documents(missing)

    file_path = tmp_path / "file.md"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        load_documents(file_path)
