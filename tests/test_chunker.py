from __future__ import annotations

from pathlib import Path

from app.models.schema import ContentType
from ingestion.chunker import DocumentMetadata, build_chunks, count_tokens
from ingestion.docx_parser import parse_docx
from ingestion.structure_parser import extract_structural_elements
from tests.docx_fixtures import build_sample_docx

DOC_META = DocumentMetadata(
    spec_number="24.501",
    document_type="TS",
    series="24",
    release="Rel-18",
    release_number=18,
    version="18.9.0",
    title="Non-Access-Stratum (NAS) protocol for 5G System (5GS)",
    source_file="24501-i90.docx",
    source_url="https://www.3gpp.org/ftp/specs/latest/Rel-18/24_series/",
)


def _chunks(tmp_path: Path):
    docx_path = tmp_path / "sample.docx"
    build_sample_docx(docx_path)
    elements = extract_structural_elements(parse_docx(docx_path))
    return build_chunks(elements, DOC_META)


def test_chunks_are_valid_schema_instances(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.spec_number == "24.501"
        assert chunk.release == "Rel-18"


def test_table_chunk_is_never_merged_with_surrounding_prose(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path)
    table_chunks = [c for c in chunks if c.is_table]

    assert len(table_chunks) == 1
    # The table chunk's text must be exactly the rendered table, not mixed
    # with paragraph prose from the same clause.
    assert table_chunks[0].text.startswith("| IE |")
    assert "shall" not in table_chunks[0].text.lower()


def test_asn1_chunk_is_isolated_and_flagged(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path)
    asn1_chunks = [c for c in chunks if c.is_asn1]

    assert len(asn1_chunks) == 1
    assert asn1_chunks[0].content_type == ContentType.ASN1
    assert "SEQUENCE" in asn1_chunks[0].text


def test_chunk_ids_follow_expected_convention(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path)
    procedure_chunk = next(c for c in chunks if "Step 1" in c.text)

    assert procedure_chunk.chunk_id.startswith("24.501_R18_18.9.0_5.1.1_")


def test_annex_chunks_flagged(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path)
    annex_chunks = [c for c in chunks if c.is_annex]
    assert len(annex_chunks) >= 1
    assert all(c.clause_number == "A" for c in annex_chunks)


def test_small_clause_produces_single_chunk_no_parent(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path)
    overview_chunks = [c for c in chunks if c.clause_number == "5.1"]

    assert len(overview_chunks) == 1
    assert overview_chunks[0].parent_chunk_id is None


def test_large_clause_splits_into_parent_and_children() -> None:
    # Build a clause with enough content to exceed the 800-token packing
    # threshold and force a parent+child split.
    from ingestion.structure_parser import StructuralElement

    long_paragraph_text = "The UE shall perform this action according to the specification. " * 40  # ~600 tokens
    elements = []
    for i in range(4):
        elements.append(
            StructuralElement(
                content_type=ContentType.PARAGRAPH,
                text=f"{long_paragraph_text} Paragraph index {i}.",
                clause_number="6.1",
                clause_title="Large clause",
                clause_path=["6", "6.1"],
                parent_clause="6",
                parent_title="Root",
            )
        )

    chunks = build_chunks(elements, DOC_META)
    parent = [c for c in chunks if c.chunk_id.endswith("_PARENT")]
    children = [c for c in chunks if c.parent_chunk_id is not None]

    assert len(parent) == 1
    assert len(children) >= 2
    assert all(c.parent_chunk_id == parent[0].chunk_id for c in children)


def test_token_counts_are_positive_and_reasonable(tmp_path: Path) -> None:
    chunks = _chunks(tmp_path)
    for chunk in chunks:
        assert chunk.token_count > 0
        assert chunk.token_count == count_tokens(chunk.text)


def test_content_hash_is_deterministic(tmp_path: Path) -> None:
    chunks_a = _chunks(tmp_path)
    chunks_b = _chunks(tmp_path)
    assert [c.content_hash for c in chunks_a] == [c.content_hash for c in chunks_b]
