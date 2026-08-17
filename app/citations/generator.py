"""Citation generation.

Citation content always comes from retrieved chunk payload metadata, never
from the LLM. The model references evidence only by its [E<n>] tag; this
module maps that tag back to a full citation string.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.schema import Chunk


@dataclass(frozen=True)
class Citation:
    tag: str  # e.g. "E1"
    spec_number: str
    release: str
    version: str
    clause_number: str
    clause_title: str
    source_locator: str
    chunk_id: str


def generate_citations(chunks: list[Chunk]) -> list[Citation]:
    """One citation per retrieved chunk, in the same [E<n>] order used when
    the evidence block was built for the prompt.
    """
    return [
        Citation(
            tag=f"E{i}",
            spec_number=chunk.spec_number,
            release=chunk.release,
            version=chunk.version,
            clause_number=chunk.clause_number,
            clause_title=chunk.clause_title,
            source_locator=chunk.source_locator,
            chunk_id=chunk.chunk_id,
        )
        for i, chunk in enumerate(chunks, start=1)
    ]


def format_citation(citation: Citation) -> str:
    return (
        f"TS {citation.spec_number} {citation.release} v{citation.version}, "
        f"Clause {citation.clause_number or 'N/A'}"
        + (f" ({citation.clause_title})" if citation.clause_title else "")
    )
