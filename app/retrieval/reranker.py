"""Cross-encoder reranking of fused hybrid candidates.

Dense+sparse retrieval optimizes for recall over a broad candidate set;
the reranker then re-scores each (query, chunk) pair jointly for
precision. It operates only on the candidate set already returned by
fusion, never the full corpus.

As with the embedding provider, the real `sentence-transformers` import is
deferred so the retrieval stack can be tested with `FakeReranker` without
downloading cross-encoder weights.
"""

from __future__ import annotations

from typing import Protocol

from loguru import logger

from app.retrieval.dense import RetrievedChunk
from app.retrieval.device import get_device


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]: ...


class CrossEncoderReranker:
    """Production reranker backed by a `sentence-transformers` CrossEncoder
    (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`).

    Device handling is automatic: CUDA when a usable GPU is present,
    otherwise CPU.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import (
                CrossEncoder,
            )
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        device = get_device()
        logger.info(
            "Loading reranker model ({}) on {}...",
            model_name,
            device,
        )
        self._model = CrossEncoder(model_name, device=device)
        logger.info("Reranker model loaded on {}", device)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        reranked = [
            RetrievedChunk(chunk=c.chunk, score=float(score))
            for c, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]


class FakeReranker:
    """Deterministic, dependency-free reranker for tests: scores by word
    overlap between query and chunk text. Not semantically meaningful, but
    exercises the full pipeline shape (sort + truncate to top_k).
    """

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        query_terms = set(query.lower().split())
        rescored = []
        for candidate in candidates:
            chunk_terms = set(candidate.chunk.text.lower().split())
            overlap = len(query_terms & chunk_terms)
            rescored.append(RetrievedChunk(chunk=candidate.chunk, score=float(overlap)))
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
