from __future__ import annotations

import pytest

from ingestion.validator import (
    ExpectedDocument,
    VersionValidationError,
    compute_sha256,
    parse_version_from_filename,
    validate_document,
)


def test_parse_version_from_filename() -> None:
    letter, major, minor = parse_version_from_filename("24501-i90.docx")
    assert (letter, major, minor) == ("i", 9, 0)


def test_parse_version_from_filename_rejects_malformed_name() -> None:
    with pytest.raises(VersionValidationError):
        parse_version_from_filename("not-a-3gpp-filename.docx")


def test_compute_sha256_matches_hashlib(tmp_path) -> None:
    import hashlib

    path = tmp_path / "sample.txt"
    path.write_bytes(b"hello 3gpp")
    expected = hashlib.sha256(b"hello 3gpp").hexdigest()
    assert compute_sha256(path) == expected


def test_validate_document_success(tmp_path) -> None:
    path = tmp_path / "24501-i90.docx"
    path.write_bytes(b"fake docx content")
    expected = ExpectedDocument(spec_number="24.501", release="Rel-18", release_number=18)

    validated = validate_document("24501-i90.docx", path, expected)

    assert validated.spec_number == "24.501"
    assert validated.version == "18.9.0"
    assert len(validated.sha256) == 64


def test_validate_document_rejects_spec_mismatch(tmp_path) -> None:
    path = tmp_path / "23501-i90.docx"
    path.write_bytes(b"content")
    expected = ExpectedDocument(spec_number="24.501", release="Rel-18", release_number=18)

    with pytest.raises(VersionValidationError, match="Spec number mismatch"):
        validate_document("23501-i90.docx", path, expected)


def test_validate_document_rejects_release_mismatch(tmp_path) -> None:
    # "h" is Rel-17's letter; expecting Rel-18 must fail.
    path = tmp_path / "24501-h90.docx"
    path.write_bytes(b"content")
    expected = ExpectedDocument(spec_number="24.501", release="Rel-18", release_number=18)

    with pytest.raises(VersionValidationError, match="Release mismatch"):
        validate_document("24501-h90.docx", path, expected)


def test_validate_document_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "24501-i90.docx"
    path.write_bytes(b"")
    expected = ExpectedDocument(spec_number="24.501", release="Rel-18", release_number=18)

    with pytest.raises(VersionValidationError, match="missing or empty"):
        validate_document("24501-i90.docx", path, expected)
