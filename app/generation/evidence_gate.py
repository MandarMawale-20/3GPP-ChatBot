"""Evidence sufficiency gate.

Runs before generation and answers one question: does the retrieved
evidence contain enough explicit information to answer this query? It
never attempts to answer the query itself — that stays with the LLM
generation step, and only once this gate passes.

The threshold-based check is the default (fast, deterministic). The
threshold is a config value so it can be tuned against the evaluation
dataset without a code change. An optional LLM-based secondary check is
provided for cases where a score threshold alone is too blunt.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from app.generation.llm import LLMProvider
from app.retrieval.dense import RetrievedChunk


@dataclass(frozen=True)
class EvidenceGateResult:
    sufficient: bool
    reason: str


class EvidenceGate:
    def __init__(self, score_threshold: float, min_chunks: int = 1) -> None:
        self._score_threshold = score_threshold
        self._min_chunks = min_chunks

    def check(self, retrieved: list[RetrievedChunk]) -> EvidenceGateResult:
        if len(retrieved) < self._min_chunks:
            return EvidenceGateResult(sufficient=False, reason="No evidence retrieved")

        top_score = max(r.score for r in retrieved)
        if top_score < self._score_threshold:
            logger.info(
                "Evidence gate: top score {:.4f} below threshold {:.4f} — abstaining",
                top_score,
                self._score_threshold,
            )
            return EvidenceGateResult(
                sufficient=False,
                reason=f"Top retrieval score {top_score:.4f} below threshold {self._score_threshold:.4f}",
            )

        return EvidenceGateResult(sufficient=True, reason="Evidence meets score threshold")


_SUFFICIENCY_CHECK_PROMPT = """You are checking whether the evidence below contains
enough EXPLICIT information to answer the question. Do not answer the
question. Respond with exactly one word: YES or NO.

Evidence:
{evidence}

Question: {query}

Does the evidence explicitly contain enough information to answer this
question? Respond YES or NO only."""


def llm_sufficiency_check(llm_provider: LLMProvider, query: str, evidence_text: str) -> EvidenceGateResult:
    """Secondary, LLM-based sufficiency check.

    Deliberately a separate, minimal prompt (no system prompt, no
    evidence-formatting reuse from generation) so this call structurally
    cannot slip into answering the question — it can only say yes/no.
    """
    prompt = _SUFFICIENCY_CHECK_PROMPT.format(evidence=evidence_text, query=query)
    response = llm_provider.generate(system_prompt="", user_prompt=prompt).strip().upper()
    sufficient = response.startswith("YES")
    return EvidenceGateResult(sufficient=sufficient, reason=f"LLM sufficiency check responded: {response!r}")
