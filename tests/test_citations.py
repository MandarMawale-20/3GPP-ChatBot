from __future__ import annotations

from app.citations.generator import format_citation, generate_citations
from app.citations.validator import extract_citation_tags, validate_citations
from app.models.schema import Chunk, ContentType


def _chunk(chunk_id: str, clause: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        spec_number="24.501",
        series="24",
        release="Rel-18",
        release_number=18,
        version="18.9.0",
        title="Test spec",
        clause_number=clause,
        clause_title="Registration procedure",
        clause_path=["5", "5.5", clause],
        content_type=ContentType.PARAGRAPH,
        text="The UE shall send a REGISTRATION REQUEST.",
        token_count=10,
        chunk_index=0,
        source_file="test.docx",
        source_url="https://example.com",
        source_locator=f"TS 24.501 v18.9.0, Clause {clause}",
        content_hash="sha256:" + "d" * 64,
    )


def test_generate_citations_tags_are_sequential() -> None:
    chunks = [_chunk("c1", "5.5.1"), _chunk("c2", "5.5.2")]
    citations = generate_citations(chunks)

    assert [c.tag for c in citations] == ["E1", "E2"]


def test_citation_content_comes_from_chunk_metadata_not_llm() -> None:
    citations = generate_citations([_chunk("c1", "5.5.1")])
    assert citations[0].spec_number == "24.501"
    assert citations[0].clause_number == "5.5.1"
    assert citations[0].version == "18.9.0"


def test_format_citation_produces_readable_string() -> None:
    citations = generate_citations([_chunk("c1", "5.5.1")])
    formatted = format_citation(citations[0])
    assert "TS 24.501" in formatted
    assert "5.5.1" in formatted
    assert "Rel-18" in formatted


def test_extract_citation_tags_finds_all_tags() -> None:
    answer = "The UE sends a request [E1]. The AMF responds [E2]."
    assert extract_citation_tags(answer) == ["E1", "E2"]


def test_validate_citations_accepts_valid_tags() -> None:
    citations = generate_citations([_chunk("c1", "5.5.1"), _chunk("c2", "5.5.2")])
    answer = "The UE sends REGISTRATION REQUEST [E1], then AMF authenticates [E2]."

    result = validate_citations(answer, citations)

    assert result.valid is True
    assert result.invalid_tags == []


def test_validate_citations_rejects_invented_tag() -> None:
    citations = generate_citations([_chunk("c1", "5.5.1")])
    # [E5] does not correspond to any retrieved evidence.
    answer = "The UE sends a request [E5]."

    result = validate_citations(answer, citations)

    assert result.valid is False
    assert "E5" in result.invalid_tags


def test_validate_citations_with_no_tags_is_valid() -> None:
    citations = generate_citations([_chunk("c1", "5.5.1")])
    result = validate_citations("No citations here.", citations)
    assert result.valid is True
