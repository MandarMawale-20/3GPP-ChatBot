"""Qdrant collection lifecycle and chunk ingestion (SRD §24/§25).

One collection (`3gpp_standards`) holds both a named dense vector and a
named sparse vector per point, plus the full `Chunk` payload — this is
what makes hybrid dense+sparse retrieval with metadata filtering possible
without juggling two separate stores (SRD §24: "preferable to maintaining
separate unrelated vector stores").
"""

from __future__ import annotations

import uuid

from loguru import logger
from qdrant_client import QdrantClient, models

from app.models.schema import Chunk
from app.retrieval.embeddings import BGE_M3_DENSE_DIM, EmbeddingProvider

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Payload fields that get a dedicated Qdrant index for fast metadata
# filtering (SRD §25). Filtering on `release`/`spec_number` before vector
# search is what prevents accidental cross-release/cross-document answers.
_PAYLOAD_INDEX_FIELDS: dict[str, models.PayloadSchemaType] = {
    "spec_number": models.PayloadSchemaType.KEYWORD,
    "release": models.PayloadSchemaType.KEYWORD,
    "version": models.PayloadSchemaType.KEYWORD,
    "document_type": models.PayloadSchemaType.KEYWORD,
    "series": models.PayloadSchemaType.KEYWORD,
    "clause_number": models.PayloadSchemaType.KEYWORD,
    "content_type": models.PayloadSchemaType.KEYWORD,
    "is_normative": models.PayloadSchemaType.BOOL,
    "is_annex": models.PayloadSchemaType.BOOL,
}

# Namespace for deterministic point IDs derived from chunk_id — Qdrant
# requires point IDs to be an unsigned int or a UUID, but our chunk_id is
# a human-readable string, so we derive a stable UUID5 from it. Using a
# fixed namespace means re-ingesting the same chunk_id always produces the
# same point ID (idempotent upserts) rather than creating duplicates.
_POINT_ID_NAMESPACE = uuid.UUID("6f2c9e6a-3b7b-4d1a-9c2a-8f6a1e2b7c11")


def chunk_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))


def retrieve_existing_hashes(client: QdrantClient, collection_name: str) -> dict[str, str]:
    """Map of chunk_id -> content_hash for points already in the collection.

    Used by `upsert_chunks` to skip chunks whose content already matches,
    avoiding redundant embedding + vector writes when re-ingesting a corpus
    that only partially changed.

    Reads only the `chunk_id` and `content_hash` payload fields (the
    `with_payload` projection) so the scan stays cheap even for large
    collections.

    Scroll is paginated via `offset`: Qdrant returns at most `limit` points
    per call, so a collection larger than 10k points would otherwise be
    truncated to its first batch. Without pagination, `skip_unchanged` would
    silently fail for every point beyond 10k — re-embedding and re-upserting
    most of the corpus on each ingest run. This matters now that the corpus
    spans multiple releases and can exceed 10k chunks.
    """
    existing: dict[str, str] = {}
    if not client.collection_exists(collection_name):
        return existing

    offset: int | None = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            with_payload=["chunk_id", "content_hash"],
            with_vectors=False,
            limit=10_000,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            content_hash = payload.get("content_hash")
            if chunk_id is not None:
                existing[chunk_id] = content_hash
        if next_offset is None:
            break
        offset = next_offset
    return existing


def get_client(url: str, api_key: str = "") -> QdrantClient:
    if url == ":memory:":
        # Used by tests — an embedded, in-process Qdrant instance with no
        # server required.
        return QdrantClient(location=":memory:")
    return QdrantClient(url=url, api_key=api_key or None)


def ensure_collection(client: QdrantClient, collection_name: str, dense_dim: int = BGE_M3_DENSE_DIM) -> None:
    """Create the collection (with dense + sparse named vectors) if it
    doesn't already exist, then ensure every required payload index is
    present. Idempotent — safe to call on every ingestion run.
    """
    if not client.collection_exists(collection_name):
        logger.info("Creating Qdrant collection: {}", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                ),
            },
        )

    for field_name, schema_type in _PAYLOAD_INDEX_FIELDS.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
        except Exception as exc:  # noqa: BLE001 — index-already-exists is not fatal
            logger.debug("Payload index for {} not (re)created: {}", field_name, exc)


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[Chunk],
    embedding_provider: EmbeddingProvider,
    batch_size: int = 16,
    skip_unchanged: bool = True,
) -> int:
    """Embed and upsert a batch of chunks into Qdrant.

    Batches the embedding calls (not just the upsert) since that's the
    expensive step — embedding one chunk at a time would be needlessly
    slow for a corpus with thousands of chunks (SRD §21/§23 scale).

    When `skip_unchanged` is True (default), chunks whose `content_hash`
    already matches the point stored under the same `chunk_id` are skipped
    entirely — no re-embedding and no re-upsert. This makes re-ingestion of
    a partially-updated corpus cheap: only genuinely new or edited chunks are
    re-processed. (The deterministic `chunk_point_id` guarantees the same
    chunk_id maps to the same Qdrant point, so an overwrite is safe.)
    """
    existing_hashes: dict[str, str] = {}
    if skip_unchanged:
        existing_hashes = retrieve_existing_hashes(client, collection_name)

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        if skip_unchanged:
            to_embed: list[Chunk] = []
            for candidate in batch:
                prior = existing_hashes.get(candidate.chunk_id)
                if prior is not None and prior == candidate.content_hash:
                    continue
                to_embed.append(candidate)
        else:
            to_embed = batch

        if not to_embed:
            continue

        texts = [_embedding_text(c) for c in to_embed]

        dense_vectors = embedding_provider.embed_dense(texts)
        sparse_vectors = embedding_provider.embed_sparse(texts)

        points = [
            models.PointStruct(
                id=chunk_point_id(chunk.chunk_id),
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: models.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
                },
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, dense_vec, sparse_vec in zip(to_embed, dense_vectors, sparse_vectors)
        ]
        client.upsert(collection_name=collection_name, points=points)
        total += len(points)
        logger.info("Upserted {}/{} chunks", total, len(chunks))

    return total


def _embedding_text(chunk: Chunk) -> str:
    """Context-enriched embedding input (SRD §20) — distinct from
    `chunk.text`, which stays the clean retrieval-facing source text
    shown as evidence. Prefixing with document/clause context measurably
    helps a bare chunk like "The UE shall send..." retrieve correctly
    against queries that mention the spec or clause explicitly.
    """
    return (
        f"3GPP TS {chunk.spec_number}\n"
        f"Release {chunk.release_number}\n"
        f"Clause {chunk.clause_number} — {chunk.clause_title}\n\n"
        f"{chunk.text}"
    )
