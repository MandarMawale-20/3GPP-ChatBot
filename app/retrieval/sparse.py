"""Sparse lexical retrieval against Qdrant.

Sparse retrieval catches exact technical identifiers (T3510, N2, 5QI,
message names) that a dense embedding can blur together with
semantically-similar-but-wrong terms.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from app.models.schema import Chunk
from app.retrieval.dense import RetrievedChunk
from app.retrieval.embeddings import SparseVector
from app.retrieval.qdrant_store import SPARSE_VECTOR_NAME


def sparse_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: SparseVector,
    top_k: int,
    query_filter: models.Filter | None = None,
) -> list[RetrievedChunk]:
    result = client.query_points(
        collection_name=collection_name,
        query=models.SparseVector(indices=query_vector.indices, values=query_vector.values),
        using=SPARSE_VECTOR_NAME,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    return [
        RetrievedChunk(chunk=Chunk.model_validate(point.payload), score=point.score)
        for point in result.points
    ]
