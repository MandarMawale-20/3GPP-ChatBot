"""Post-generation claim/evidence verification.

Runs after the LLM produces an answer, checking it against the evidence
that was actually supplied — this catches an answer that drifted from its
sources even though the evidence gate allowed generation to proceed.

Citation integrity is delegated to `app/citations/validator.py`; this
module focuses on evidence coverage and unsupported numeric/technical
values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.schema import Chunk

# Technical identifiers/numeric values that must be traceable back to the
# evidence verbatim if they appear in the answer — 3GPP timers (T3510),
# percentages, and bare numbers are exactly the kind of detail an LLM is
# prone to inventing with high confidence.
_TIMER_RE = re.compile(r"\bT\d{4}\b")
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?%")
_NUMERIC_TOKEN_RE = re.compile(r"\b\d{2,}\b")  # 2+ digit bare numbers (avoids flagging single digits like "5G")


@dataclass
class VerificationResult:
    passed: bool
    unsupported_claims: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _evidence_corpus_text(chunks: list[Chunk]) -> str:
    return "\n".join(c.text for c in chunks)


def check_numeric_claims_supported(answer: str, chunks: list[Chunk]) -> list[str]:
    """Flag any timer/percentage/numeric token in the answer that does not
    appear verbatim anywhere in the retrieved evidence text.

    A precision-favoring heuristic, not a proof of correctness — a number
    appearing in the evidence doesn't guarantee it was used in the right
    context, but a number absent from the evidence is an unambiguous sign
    of invention.
    """
    evidence_text = _evidence_corpus_text(chunks)
    unsupported: list[str] = []

    for pattern in (_TIMER_RE, _PERCENT_RE, _NUMERIC_TOKEN_RE):
        for match in pattern.findall(answer):
            if match not in evidence_text and match not in unsupported:
                unsupported.append(match)

    return unsupported


def check_evidence_coverage(answer: str, num_evidence_blocks: int) -> list[str]:
    """Warn if the answer makes substantive claims (i.e. isn't just the
    abstention message) without citing any evidence tag at all.
    """
    from app.generation.prompts import ABSTENTION_MESSAGE

    warnings: list[str] = []
    if answer.strip() == ABSTENTION_MESSAGE:
        return warnings

    citation_tags = re.findall(r"\[E(\d+)\]", answer)
    if num_evidence_blocks > 0 and not citation_tags:
        warnings.append("Answer contains no evidence citations despite evidence being available")
    return warnings


def verify_answer(answer: str, chunks: list[Chunk]) -> VerificationResult:
    """Run the full post-generation verification pass."""
    from app.generation.prompts import ABSTENTION_MESSAGE

    if answer.strip() == ABSTENTION_MESSAGE:
        # An abstention always passes verification — there is nothing to
        # verify, and abstention is the safe outcome.
        return VerificationResult(passed=True)

    unsupported = check_numeric_claims_supported(answer, chunks)
    warnings = check_evidence_coverage(answer, len(chunks))

    # A hard failure (unsupported numeric/technical claim) means the
    # caller should discard the answer and abstain instead — see
    # app/generation/generator.py.
    passed = len(unsupported) == 0

    return VerificationResult(passed=passed, unsupported_claims=unsupported, warnings=warnings)
