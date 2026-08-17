"""Hybrid retrieval: metadata filtering + RRF fusion of dense and sparse results.

RRF (Reciprocal Rank Fusion) is implemented directly in Python rather than
relying on a Qdrant server-side fusion API. This keeps fusion behavior
stable across Qdrant versions and easy to unit test in isolation.
"""

from __future__ import annotations

from qdrant_client import models

from app.retrieval.dense import RetrievedChunk

# Standard RRF damping constant (k=60 is the conventional default from the
# original RRF paper and what most vector DB implementations use).
RRF_K = 60


def build_release_spec_filter(release: str, spec_number: str | None = None) -> models.Filter:
    """Metadata filter applied before vector search — this is the mechanism
    that makes release/spec isolation a retrieval guarantee rather than a hope.
    """
    conditions: list[models.FieldCondition] = [
        models.FieldCondition(key="release", match=models.MatchValue(value=release))
    ]
    if spec_number:
        conditions.append(models.FieldCondition(key="spec_number", match=models.MatchValue(value=spec_number)))
    # qdrant_client's typing expects a broader condition union; the cast is safe at runtime.
    return models.Filter(must=conditions)  # type: ignore[arg-type]


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse multiple ranked result lists into one, by chunk_id.

    Each list contributes `1 / (k + rank)` to a chunk's fused score
    (rank is 1-indexed). A chunk appearing near the top of either the
    dense or sparse list scores highly.
    """
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, retrieved in enumerate(results, start=1):
            chunk_id = retrieved.chunk.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            # Keep the highest-scoring occurrence's chunk object (payload
            # is identical either way, but this avoids relying on dict
            # iteration order for which copy wins).
            if chunk_id not in chunk_by_id or retrieved.score > chunk_by_id[chunk_id].score:
                chunk_by_id[chunk_id] = retrieved

    fused = [
        RetrievedChunk(chunk=chunk_by_id[chunk_id].chunk, score=score)
        for chunk_id, score in fused_scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused
