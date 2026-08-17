"""Grounded generation orchestrator.

Ties together the evidence gate, LLM generation, post-generation
verification, and citation validation into the single flow the API calls.
Abstention is treated as a normal, successful outcome at every stage, not
an error path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from app.citations.generator import Citation, generate_citations
from app.citations.validator import validate_citations
from app.generation.evidence_gate import EvidenceGate
from app.generation.llm import LLMProvider
from app.generation.prompts import ABSTENTION_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from app.generation.verifier import verify_answer
from app.retrieval.dense import RetrievedChunk


@dataclass
class ChatResult:
    answer: str
    abstained: bool
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    abstain_reason: str = ""


class GroundedGenerator:
    def __init__(self, llm_provider: LLMProvider, evidence_gate: EvidenceGate) -> None:
        self._llm_provider = llm_provider
        self._evidence_gate = evidence_gate

    def answer(self, query: str, retrieved: list[RetrievedChunk]) -> ChatResult:
        gate_result = self._evidence_gate.check(retrieved)
        if not gate_result.sufficient:
            logger.info("Insufficient evidence — abstaining ({})", gate_result.reason)
            return ChatResult(answer=ABSTENTION_MESSAGE, abstained=True, abstain_reason=gate_result.reason)

        chunks = [r.chunk for r in retrieved]
        citations = generate_citations(chunks)

        user_prompt = build_user_prompt(query, chunks)
        raw_answer = self._llm_provider.generate(SYSTEM_PROMPT, user_prompt)

        if raw_answer.strip() == ABSTENTION_MESSAGE:
            return ChatResult(answer=ABSTENTION_MESSAGE, abstained=True, abstain_reason="Model chose to abstain")

        citation_check = validate_citations(raw_answer, citations)
        if not citation_check.valid:
            logger.warning(
                "Citation validation failed — invalid tags: {} — abstaining",
                citation_check.invalid_tags,
            )
            return ChatResult(
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                abstain_reason=f"Invalid citation tags: {citation_check.invalid_tags}",
            )

        verification = verify_answer(raw_answer, chunks)
        if not verification.passed:
            logger.warning(
                "Claim verification failed — unsupported claims: {} — abstaining",
                verification.unsupported_claims,
            )
            return ChatResult(
                answer=ABSTENTION_MESSAGE,
                abstained=True,
                abstain_reason=f"Unsupported claims: {verification.unsupported_claims}",
            )

        confidence = min(r.score for r in retrieved) if retrieved else 0.0
        used_citations = [c for c in citations if c.tag in citation_check.used_tags]

        return ChatResult(
            answer=raw_answer,
            abstained=False,
            citations=used_citations or citations,
            confidence=confidence,
        )
