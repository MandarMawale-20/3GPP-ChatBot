"""Citation validation.

Every citation tag the LLM outputs must correspond to a chunk that was
actually retrieved and supplied as evidence. A tag outside that range is
either a hallucinated reference or a format error, so the answer is
rejected rather than trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.citations.generator import Citation

_CITATION_TAG_RE = re.compile(r"\[E(\d+)\]")


@dataclass
class CitationValidationResult:
    valid: bool
    used_tags: list[str] = field(default_factory=list)
    invalid_tags: list[str] = field(default_factory=list)


def extract_citation_tags(answer: str) -> list[str]:
    return [f"E{n}" for n in _CITATION_TAG_RE.findall(answer)]


def validate_citations(answer: str, citations: list[Citation]) -> CitationValidationResult:
    """Reject any [E<n>] tag in the answer that doesn't match a retrieved citation."""
    valid_tags = {c.tag for c in citations}
    used_tags = extract_citation_tags(answer)

    invalid_tags = [tag for tag in used_tags if tag not in valid_tags]

    return CitationValidationResult(
        valid=len(invalid_tags) == 0,
        used_tags=used_tags,
        invalid_tags=invalid_tags,
    )
