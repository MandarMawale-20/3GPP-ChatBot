"""Orchestrates the full retrieval pipeline:

    metadata filter -> dense + sparse search -> RRF fusion -> reranking

This is the single entry point the API/evidence-gate layer calls; it does
not know or care whether the embedding/reranker backends are the real
models or test doubles.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from qdrant_client import QdrantClient

from app.retrieval.dense import RetrievedChunk, dense_search
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.hybrid import build_release_spec_filter, reciprocal_rank_fusion
from app.retrieval.query_preprocessor import extract_query_filters
from app.retrieval.reranker import Reranker
from app.retrieval.sparse import sparse_search


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_k: int = 8


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self._config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        release: str | None = None,
        spec_number: str | None = None,
        rerank_top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Run the full filter -> dense+sparse -> RRF -> rerank pipeline.

        `rerank_top_k` optionally overrides the configured default for a
        single call (used by the /search debug endpoint) without mutating
        shared retriever state — the retriever instance is a cached
        singleton reused across concurrent requests, so per-call overrides
        must never touch `self._config`.

        Spec-number scoping: an explicit `spec_number` argument always wins
        (it is set by the frontend's document selector or an API caller). If
        none is provided, the query text is scanned for an embedded 3GPP spec
        reference (e.g. "TS 24.501") and, when found, used to scope retrieval
        to that document only.

        Release control is a hard safety boundary — a query scoped to a
        specific release never retrieves content from another release.
        When ``release`` is ``None``, the release filter is omitted, enabling
        cross-release retrieval across all indexed releases; release metadata
        remains attached to every retrieved chunk for citation purposes.
        """
        resolved_spec_number = spec_number
        if resolved_spec_number is None:
            extracted = extract_query_filters(query)
            resolved_spec_number = extracted.spec_number
            if resolved_spec_number is not None:
                logger.info(
                    "Query implies spec_number '{}' (derived from query text); scoping retrieval",
                    resolved_spec_number,
                )

        query_filter = build_release_spec_filter(release, resolved_spec_number)

        dense_vector = self._embedding_provider.embed_dense([query])[0]
        sparse_vector = self._embedding_provider.embed_sparse([query])[0]

        dense_results = dense_search(
            self._client, self._collection_name, dense_vector, self._config.dense_top_k, query_filter
        )
        sparse_results = sparse_search(
            self._client, self._collection_name, sparse_vector, self._config.sparse_top_k, query_filter
        )

        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        top_k = rerank_top_k if rerank_top_k is not None else self._config.rerank_top_k
        return self._reranker.rerank(query, fused, top_k)
