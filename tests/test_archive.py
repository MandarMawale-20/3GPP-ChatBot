from __future__ import annotations

import zipfile

import pytest

from ingestion.archive import ArchiveError, find_document_file, safe_extract


def _make_zip(path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_safe_extract_extracts_normal_archive(tmp_path) -> None:
    zip_path = tmp_path / "sample.zip"
    _make_zip(zip_path, {"24501-i90.docx": b"fake content"})
    dest = tmp_path / "extracted"

    safe_extract(zip_path, dest)

    assert (dest / "24501-i90.docx").exists()


def test_safe_extract_rejects_path_traversal(tmp_path) -> None:
    zip_path = tmp_path / "malicious.zip"
    _make_zip(zip_path, {"../../evil.txt": b"pwned"})
    dest = tmp_path / "extracted"

    with pytest.raises(ArchiveError, match="Unsafe path"):
        safe_extract(zip_path, dest)


def test_safe_extract_rejects_corrupt_zip(tmp_path) -> None:
    zip_path = tmp_path / "corrupt.zip"
    zip_path.write_bytes(b"not a real zip file")
    dest = tmp_path / "extracted"

    with pytest.raises(ArchiveError, match="Corrupt ZIP"):
        safe_extract(zip_path, dest)


def test_find_document_file_selects_largest_docx(tmp_path) -> None:
    (tmp_path / "cover_sheet.docx").write_bytes(b"x" * 10)
    (tmp_path / "24501-i90.docx").write_bytes(b"x" * 1000)

    chosen = find_document_file(tmp_path)

    assert chosen.name == "24501-i90.docx"


def test_find_document_file_ignores_temp_files(tmp_path) -> None:
    (tmp_path / "~$24501-i90.docx").write_bytes(b"x" * 5000)  # Word lock file, larger but must be ignored
    (tmp_path / "24501-i90.docx").write_bytes(b"x" * 100)

    chosen = find_document_file(tmp_path)

    assert chosen.name == "24501-i90.docx"


def test_find_document_file_raises_when_none_found(tmp_path) -> None:
    (tmp_path / "readme.txt").write_text("nothing here")
    with pytest.raises(ArchiveError):
        find_document_file(tmp_path)
