from __future__ import annotations

from app.generation.evidence_gate import EvidenceGate
from app.generation.generator import GroundedGenerator
from app.generation.llm import FakeLLMProvider
from app.generation.prompts import ABSTENTION_MESSAGE
from app.models.schema import Chunk, ContentType
from app.retrieval.dense import RetrievedChunk


def _retrieved(score: float, text: str = "The UE shall send a REGISTRATION REQUEST message to the AMF.") -> RetrievedChunk:
    chunk = Chunk(
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
        content_hash="sha256:" + "f" * 64,
    )
    return RetrievedChunk(chunk=chunk, score=score)


def test_abstains_when_evidence_gate_fails() -> None:
    llm = FakeLLMProvider(response="Should never be called")
    gate = EvidenceGate(score_threshold=0.9)
    generator = GroundedGenerator(llm_provider=llm, evidence_gate=gate)

    result = generator.answer("What is T3510?", [_retrieved(0.1)])

    assert result.abstained is True
    assert result.answer == ABSTENTION_MESSAGE


def test_returns_grounded_answer_with_valid_citation() -> None:
    llm = FakeLLMProvider(response="The UE sends a REGISTRATION REQUEST [E1].")
    gate = EvidenceGate(score_threshold=0.1)
    generator = GroundedGenerator(llm_provider=llm, evidence_gate=gate)

    result = generator.answer("What does the UE send?", [_retrieved(0.9)])

    assert result.abstained is False
    assert "[E1]" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].spec_number == "24.501"


def test_abstains_on_invented_citation_tag() -> None:
    # [E9] does not correspond to any retrieved chunk.
    llm = FakeLLMProvider(response="The UE sends a request [E9].")
    gate = EvidenceGate(score_threshold=0.1)
    generator = GroundedGenerator(llm_provider=llm, evidence_gate=gate)

    result = generator.answer("What does the UE send?", [_retrieved(0.9)])

    assert result.abstained is True
    assert "Invalid citation" in result.abstain_reason


def test_abstains_on_unsupported_numeric_claim() -> None:
    llm = FakeLLMProvider(response="The retry timer is set to 8888 seconds [E1].")
    gate = EvidenceGate(score_threshold=0.1)
    generator = GroundedGenerator(llm_provider=llm, evidence_gate=gate)

    result = generator.answer("What is the retry timer?", [_retrieved(0.9)])

    assert result.abstained is True
    assert "Unsupported claims" in result.abstain_reason


def test_model_choosing_to_abstain_is_respected() -> None:
    llm = FakeLLMProvider(response=ABSTENTION_MESSAGE)
    gate = EvidenceGate(score_threshold=0.1)
    generator = GroundedGenerator(llm_provider=llm, evidence_gate=gate)

    result = generator.answer("Some question", [_retrieved(0.9)])

    assert result.abstained is True
    assert result.answer == ABSTENTION_MESSAGE


def test_confidence_reflects_weakest_retrieved_score() -> None:
    llm = FakeLLMProvider(response="Answer here [E1].")
    gate = EvidenceGate(score_threshold=0.1)
    generator = GroundedGenerator(llm_provider=llm, evidence_gate=gate)

    result = generator.answer("Question", [_retrieved(0.9)])

    assert result.confidence == 0.9
