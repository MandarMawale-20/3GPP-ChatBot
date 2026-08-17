"""Tests for the canonical Chunk schema (app.models.schema).

These lock down the invariants the rest of the pipeline relies on:
content_type/flag consistency, hash format, table shape, release format.
Synthetic fixtures only — no dependency on real 3GPP files (SRD §24).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.schema import Chunk, ContentType, SourceDocument, TableCell, TableData

VALID_HASH = "sha256:" + "a" * 64


def _base_chunk_kwargs(**overrides) -> dict:
    kwargs = dict(
        chunk_id="24.501_R18_18.9.0_5.5.1_003",
        spec_number="24.501",
        series="24",
        release="Rel-18",
        release_number=18,
        version="18.9.0",
        title="Non-Access-Stratum (NAS) protocol for 5G System (5GS)",
        clause_number="5.5.1",
        clause_title="Registration procedure",
        clause_path=["5", "5.5", "5.5.1"],
        content_type=ContentType.PARAGRAPH,
        text="The UE shall send a REGISTRATION REQUEST message.",
        token_count=42,
        chunk_index=0,
        source_file="24501-i90.docx",
        source_url="https://www.3gpp.org/ftp/specs/latest/Rel-18/24_series/",
        source_locator="TS 24.501 v18.9.0, Clause 5.5.1, Pages 123-125",
        content_hash=VALID_HASH,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_paragraph_chunk_constructs() -> None:
    chunk = Chunk(**_base_chunk_kwargs())
    assert chunk.content_type == ContentType.PARAGRAPH
    assert chunk.is_table is False


def test_table_chunk_requires_table_data() -> None:
    with pytest.raises(ValidationError, match="requires table_data"):
        Chunk(**_base_chunk_kwargs(content_type=ContentType.TABLE, is_table=True))


def test_table_chunk_with_data_and_flag_succeeds() -> None:
    table = TableData(
        headers=["IE", "Presence"],
        rows=[[TableCell(text="5GMM cause"), TableCell(text="M")]],
    )
    chunk = Chunk(
        **_base_chunk_kwargs(
            content_type=ContentType.TABLE,
            is_table=True,
            table_data=table,
        )
    )
    assert chunk.table_data.headers == ["IE", "Presence"]


def test_content_type_flag_mismatch_rejected() -> None:
    # content_type=ASN1 but is_asn1 left False — must fail (SRD hallucination
    # of a mismatched filter would break metadata-filtered retrieval).
    with pytest.raises(ValidationError, match="is_asn1"):
        Chunk(**_base_chunk_kwargs(content_type=ContentType.ASN1, is_asn1=False))


def test_invalid_content_hash_rejected() -> None:
    with pytest.raises(ValidationError, match="content_hash"):
        Chunk(**_base_chunk_kwargs(content_hash="not-a-valid-hash"))


def test_invalid_release_format_rejected() -> None:
    with pytest.raises(ValidationError, match="release"):
        Chunk(**_base_chunk_kwargs(release="18"))


def test_table_row_width_mismatch_rejected() -> None:
    with pytest.raises(ValidationError, match="effective width"):
        TableData(
            headers=["A", "B", "C"],
            rows=[[TableCell(text="x"), TableCell(text="y")]],  # only 2 wide
        )


def test_merged_cell_colspan_satisfies_width() -> None:
    # A 2-wide merged cell plus a 1-wide cell should satisfy a 3-column header.
    table = TableData(
        headers=["A", "B", "C"],
        rows=[[TableCell(text="merged", colspan=2), TableCell(text="z")]],
    )
    assert len(table.rows[0]) == 2


def test_source_document_validates_sha256() -> None:
    with pytest.raises(ValidationError, match="sha256"):
        SourceDocument(
            spec_number="23.501",
            release="Rel-18",
            version="18.5.0",
            filename="23501-i50.docx",
            sha256="not-a-hash",
            downloaded_at="2026-08-16T00:00:00Z",
            source_url="https://www.3gpp.org/ftp/specs/latest/Rel-18/23_series/",
        )


def test_source_document_validates_release_format() -> None:
    with pytest.raises(ValidationError, match="release"):
        SourceDocument(
            spec_number="23.501",
            release="18",
            version="18.5.0",
            filename="23501-i50.docx",
            sha256="a" * 64,
            downloaded_at="2026-08-16T00:00:00Z",
            source_url="https://www.3gpp.org/ftp/specs/latest/Rel-18/23_series/",
        )


def test_chunk_json_schema_example_is_valid() -> None:
    # The example embedded in the schema (used for docs/tests elsewhere)
    # must itself pass validation, or the docs would be lying.
    example = Chunk.model_config["json_schema_extra"]["example"]
    chunk = Chunk(**example)
    assert chunk.spec_number == "24.501"
