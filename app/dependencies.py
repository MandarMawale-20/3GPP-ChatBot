"""Application dependency wiring.

Builds the (lazily-loaded, heavy) singletons — Qdrant client, embedding
provider, reranker, LLM provider — exactly once per process and hands them
to FastAPI route handlers via `Depends`. Kept separate from `app/main.py`
so the wiring can be swapped for fakes in tests without importing FastAPI
at all.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.generation.evidence_gate import EvidenceGate
from app.generation.generator import GroundedGenerator
from app.generation.llm import build_default_llm_provider
from app.retrieval.embeddings import BGEM3EmbeddingProvider, EmbeddingProvider
from app.retrieval.qdrant_store import get_client
from app.retrieval.reranker import CrossEncoderReranker, Reranker
from app.retrieval.retriever import RetrievalConfig, Retriever


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    settings = get_settings().settings
    return BGEM3EmbeddingProvider(model_name=settings.embedding_model)


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings().settings
    return CrossEncoderReranker(model_name=settings.reranker_model)


@lru_cache
def get_retriever() -> Retriever:
    settings = get_settings().settings
    client = get_client(settings.qdrant_url, settings.qdrant_api_key)
    config = RetrievalConfig(
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rerank_top_k=settings.rerank_top_k,
    )
    return Retriever(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding_provider=get_embedding_provider(),
        reranker=get_reranker(),
        config=config,
    )


@lru_cache
def get_generator() -> GroundedGenerator:
    settings = get_settings().settings
    llm_provider = build_default_llm_provider(api_key=settings.gemini_api_key, model_name=settings.llm_model)
    evidence_gate = EvidenceGate(score_threshold=settings.evidence_score_threshold)
    return GroundedGenerator(llm_provider=llm_provider, evidence_gate=evidence_gate)
