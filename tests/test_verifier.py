from __future__ import annotations

from app.generation.prompts import ABSTENTION_MESSAGE
from app.generation.verifier import (
    check_evidence_coverage,
    check_numeric_claims_supported,
    verify_answer,
)
from app.models.schema import Chunk, ContentType


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="c1",
        spec_number="24.501",
        series="24",
        release="Rel-18",
        release_number=18,
        version="18.9.0",
        title="Test spec",
        clause_number="5.5.1",
        clause_title="Registration",
        clause_path=["5", "5.5", "5.5.1"],
        content_type=ContentType.PARAGRAPH,
        text=text,
        token_count=10,
        chunk_index=0,
        source_file="test.docx",
        source_url="https://example.com",
        source_locator="TS 24.501 v18.9.0, Clause 5.5.1",
        content_hash="sha256:" + "e" * 64,
    )


def test_numeric_claim_supported_by_evidence_passes() -> None:
    chunks = [_chunk("Timer T3510 has a default value of 15 seconds.")]
    answer = "T3510 is used to control registration retries [E1]."

    unsupported = check_numeric_claims_supported(answer, chunks)

    assert unsupported == []


def test_numeric_claim_not_in_evidence_is_flagged() -> None:
    chunks = [_chunk("Timer T3510 controls registration retries.")]
    # T3999 never appears anywhere in the evidence -> must be flagged.
    answer = "Timer T3999 also affects registration [E1]."

    unsupported = check_numeric_claims_supported(answer, chunks)

    assert "T3999" in unsupported


def test_evidence_coverage_flags_missing_citations() -> None:
    warnings = check_evidence_coverage("The UE sends a request with no citation tag.", num_evidence_blocks=2)
    assert len(warnings) == 1


def test_evidence_coverage_ignores_abstention() -> None:
    warnings = check_evidence_coverage(ABSTENTION_MESSAGE, num_evidence_blocks=2)
    assert warnings == []


def test_verify_answer_passes_for_abstention() -> None:
    result = verify_answer(ABSTENTION_MESSAGE, [])
    assert result.passed is True


def test_verify_answer_fails_for_unsupported_numeric_claim() -> None:
    chunks = [_chunk("The UE shall send a REGISTRATION REQUEST message.")]
    answer = "The retry timer is set to 9999 seconds [E1]."

    result = verify_answer(answer, chunks)

    assert result.passed is False
    assert "9999" in result.unsupported_claims


def test_verify_answer_passes_for_fully_supported_claim() -> None:
    chunks = [_chunk("Timer T3510 has a default value of 15 seconds for registration retries.")]
    answer = "T3510 has a default value of 15 seconds [E1]."

    result = verify_answer(answer, chunks)

    assert result.passed is True
