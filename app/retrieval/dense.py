"""Dense (semantic) retrieval against Qdrant."""

from __future__ import annotations

from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from app.models.schema import Chunk
from app.retrieval.qdrant_store import DENSE_VECTOR_NAME


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def dense_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    query_filter: models.Filter | None = None,
) -> list[RetrievedChunk]:
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return [
        RetrievedChunk(chunk=Chunk.model_validate(point.payload), score=point.score)
        for point in result.points
    ]
