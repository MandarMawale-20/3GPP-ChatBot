from __future__ import annotations

from pathlib import Path

from ingestion.pipeline import process_local_document, read_jsonl, write_jsonl
from tests.docx_fixtures import build_sample_docx


def test_process_local_document_produces_valid_chunks(tmp_path: Path) -> None:
    docx_path = tmp_path / "24501-i90.docx"
    build_sample_docx(docx_path)

    chunks = process_local_document(
        docx_path=docx_path,
        spec_number="24.501",
        series="24",
        release="Rel-18",
        release_number=18,
        version="18.9.0",
        title="Non-Access-Stratum (NAS) protocol for 5G System (5GS)",
        source_file="24501-i90.docx",
        source_url="https://www.3gpp.org/ftp/specs/latest/Rel-18/24_series/",
    )

    assert len(chunks) > 0
    assert all(c.spec_number == "24.501" for c in chunks)
    assert any(c.is_table for c in chunks)
    assert any(c.is_asn1 for c in chunks)


def test_jsonl_round_trip_preserves_chunk_data(tmp_path: Path) -> None:
    docx_path = tmp_path / "24501-i90.docx"
    build_sample_docx(docx_path)

    chunks = process_local_document(
        docx_path=docx_path,
        spec_number="24.501",
        series="24",
        release="Rel-18",
        release_number=18,
        version="18.9.0",
        title="Non-Access-Stratum (NAS) protocol for 5G System (5GS)",
        source_file="24501-i90.docx",
        source_url="https://example.com",
    )

    jsonl_path = tmp_path / "24.501.jsonl"
    write_jsonl(chunks, jsonl_path)
    reloaded = read_jsonl(jsonl_path)

    assert len(reloaded) == len(chunks)
    assert [c.chunk_id for c in reloaded] == [c.chunk_id for c in chunks]
    assert [c.content_hash for c in reloaded] == [c.content_hash for c in chunks]


def test_jsonl_is_one_record_per_line(tmp_path: Path) -> None:
    docx_path = tmp_path / "24501-i90.docx"
    build_sample_docx(docx_path)
    chunks = process_local_document(
        docx_path=docx_path,
        spec_number="24.501",
        series="24",
        release="Rel-18",
        release_number=18,
        version="18.9.0",
        title="Title",
        source_file="24501-i90.docx",
        source_url="https://example.com",
    )

    jsonl_path = tmp_path / "out.jsonl"
    write_jsonl(chunks, jsonl_path)

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == len(chunks)
    import json

    for line in lines:
        json.loads(line)  # must be valid standalone JSON
