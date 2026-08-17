from __future__ import annotations

from app.generation.evidence_gate import EvidenceGate
from app.generation.generator import GroundedGenerator
from app.generation.llm import FakeLLMProvider
from app.generation.prompts import ABSTENTION_MESSAGE
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
        title="Test spec",
        clause_number="5.5.1",
        clause_title="Registration",
        clause_path=["5", "5.5", "5.5.1"],
        content_type=ContentType.PARAGRAPH,
        text="The UE shall send a REGISTRATION REQUEST message to the AMF.",
        token_count=10,
        chunk_index=0,
        source_file="test.docx",
        source_url="https://example.com",
        source_locator="TS 24.501 v18.9.0, Clause 5.5.1",
        content_hash="sha256:" + "f" * 64,
    )
    return RetrievedChunk(chunk=chunk, score=score)


def test_abstains_when_evidence_gate_fails() -> None:
    """Below-threshold evidence causes the generator to abstain without calling the LLM."""
    llm = FakeLLMProvider(response="Should never be called")
    gate = EvidenceGate(score_threshold=0.9)
    generator = GroundedGenerator(llm_provider=llm, evidence_gate=gate)

    result = generator.answer("What is T3510?", [_retrieved(0.1)])

    assert result.abstained is True
    assert result.answer == ABSTENTION_MESSAGE
