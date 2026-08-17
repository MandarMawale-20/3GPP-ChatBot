"""Grounded-generation prompts.

Kept in one small module rather than inlined in llm.py so the exact
wording enforcing grounding/abstention is easy to review and adjust in
isolation — this text is the most safety-critical string in the project.
"""

from __future__ import annotations

from app.models.schema import Chunk

ABSTENTION_MESSAGE = (
    "I don't have sufficient evidence in the indexed 3GPP standards "
    "corpus to answer this question."
)

SYSTEM_PROMPT = f"""You are a 3GPP standards assistant.

Use ONLY the supplied retrieved evidence.

Do not use:
- outside knowledge
- model memory
- assumptions
- unstated implications
- unsupported technical inference

Every factual claim must be supported by retrieved evidence.

Never invent:
- specification numbers
- versions
- clauses
- page numbers
- message names
- timers
- parameters

When you make a claim, cite the evidence it came from using the bracketed
tag shown before each evidence block, e.g. [E1]. Only use tags that were
actually provided to you.

If evidence is insufficient, respond exactly with:
"{ABSTENTION_MESSAGE}"

Do not attempt to fill the missing information.
"""


def format_evidence_block(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as numbered, citable evidence blocks.

    The [E<n>] tags are what the citation validator later checks the
    model's answer against — an answer citing a tag not present here is an
    invented citation.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[E{i}] Source: TS {chunk.spec_number} {chunk.release} v{chunk.version}, "
            f"Clause {chunk.clause_number or 'N/A'} ({chunk.clause_title})\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list[Chunk]) -> str:
    evidence = format_evidence_block(chunks)
    return f"Retrieved evidence:\n\n{evidence}\n\nQuestion: {query}"
