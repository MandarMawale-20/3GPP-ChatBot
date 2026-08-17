from __future__ import annotations

from app.generation.evidence_gate import EvidenceGate, llm_sufficiency_check
from app.generation.llm import FakeLLMProvider
from app.models.schema import Chunk, ContentType
from app.retrieval.dense import RetrievedChunk


def _retrieved(score: float) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="c1",
        spec_number="24.501",
        series="24",
        release="Rel-18",
        release_number=18,
        version="18.9.0",
        title="Test",
        clause_number="5.5.1",
        clause_title="Registration",
        clause_path=["5", "5.5", "5.5.1"],
        content_type=ContentType.PARAGRAPH,
        text="The UE shall send a REGISTRATION REQUEST.",
        token_count=10,
        chunk_index=0,
        source_file="test.docx",
        source_url="https://example.com",
        source_locator="TS 24.501 v18.9.0, Clause 5.5.1",
        content_hash="sha256:" + "c" * 64,
    )
    return RetrievedChunk(chunk=chunk, score=score)


def test_gate_rejects_empty_evidence() -> None:
    gate = EvidenceGate(score_threshold=0.35)
    result = gate.check([])
    assert result.sufficient is False


def test_gate_rejects_below_threshold_score() -> None:
    gate = EvidenceGate(score_threshold=0.5)
    result = gate.check([_retrieved(0.2)])
    assert result.sufficient is False


def test_gate_accepts_above_threshold_score() -> None:
    gate = EvidenceGate(score_threshold=0.5)
    result = gate.check([_retrieved(0.9)])
    assert result.sufficient is True


def test_gate_uses_max_score_among_candidates() -> None:
    gate = EvidenceGate(score_threshold=0.5)
    result = gate.check([_retrieved(0.1), _retrieved(0.9)])
    assert result.sufficient is True


def test_llm_sufficiency_check_parses_yes() -> None:
    llm = FakeLLMProvider(response="YES")
    result = llm_sufficiency_check(llm, "What is T3510?", "T3510 is a registration timer.")
    assert result.sufficient is True


def test_llm_sufficiency_check_parses_no() -> None:
    llm = FakeLLMProvider(response="NO")
    result = llm_sufficiency_check(llm, "What is the CEO's salary?", "Irrelevant evidence.")
    assert result.sufficient is False
