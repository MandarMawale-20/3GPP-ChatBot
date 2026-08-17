from __future__ import annotations

from app.models.schema import Chunk, ContentType
from app.retrieval.dense import RetrievedChunk, dense_search
from app.retrieval.embeddings import FakeEmbeddingProvider
from app.retrieval.hybrid import build_release_spec_filter, reciprocal_rank_fusion
from app.retrieval.qdrant_store import ensure_collection, get_client, upsert_chunks
from app.retrieval.reranker import FakeReranker
from app.retrieval.retriever import RetrievalConfig, Retriever
from app.retrieval.sparse import sparse_search

COLLECTION = "retrieval_test"


def _make_chunk(chunk_id: str, text: str, spec_number: str, release: str = "Rel-18", clause: str = "5.5.1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        spec_number=spec_number,
        series=spec_number.split(".")[0],
        release=release,
        release_number=int(release.split("-")[1]),
        version="18.9.0",
        title="Test spec",
        clause_number=clause,
        clause_title="Registration procedure",
        clause_path=["5", "5.5", clause],
        content_type=ContentType.PARAGRAPH,
        text=text,
        token_count=10,
        chunk_index=0,
        source_file="test.docx",
        source_url="https://example.com",
        source_locator=f"TS {spec_number} v18.9.0, Clause {clause}",
        content_hash="sha256:" + "b" * 64,
    )


def _seeded_client(collection: str):
    client = get_client(":memory:")
    ensure_collection(client, collection, dense_dim=8)
    provider = FakeEmbeddingProvider(dense_dim=8)

    chunks = [
        _make_chunk("c1", "The UE shall send a REGISTRATION REQUEST to the AMF.", "24.501"),
        _make_chunk("c2", "T3510 timer controls registration retry behavior.", "24.501"),
        _make_chunk("c3", "The AMF selects an appropriate SMF for the session.", "23.501"),
        _make_chunk("c4", "This document is Release 17 content.", "24.501", release="Rel-17"),
    ]
    upsert_chunks(client, collection, chunks, provider)
    return client, provider


def test_metadata_filter_isolates_release() -> None:
    client, provider = _seeded_client(COLLECTION + "_filter")
    query_filter = build_release_spec_filter("Rel-18")

    query_vector = provider.embed_dense(["registration"])[0]
    results = dense_search(client, COLLECTION + "_filter", query_vector, top_k=10, query_filter=query_filter)

    assert all(r.chunk.release == "Rel-18" for r in results)
    assert not any(r.chunk.chunk_id == "c4" for r in results)


def test_metadata_filter_isolates_spec_number() -> None:
    client, provider = _seeded_client(COLLECTION + "_spec_filter")
    query_filter = build_release_spec_filter("Rel-18", spec_number="23.501")

    query_vector = provider.embed_dense(["registration"])[0]
    results = dense_search(client, COLLECTION + "_spec_filter", query_vector, top_k=10, query_filter=query_filter)

    assert all(r.chunk.spec_number == "23.501" for r in results)


def test_sparse_search_finds_exact_identifier() -> None:
    client, provider = _seeded_client(COLLECTION + "_sparse")
    query_filter = build_release_spec_filter("Rel-18")
    sparse_vector = provider.embed_sparse(["T3510 timer"])[0]

    results = sparse_search(client, COLLECTION + "_sparse", sparse_vector, top_k=10, query_filter=query_filter)

    assert any(r.chunk.chunk_id == "c2" for r in results)


def test_reciprocal_rank_fusion_favors_items_ranked_highly_in_either_list() -> None:
    chunk_a = _make_chunk("a", "text a", "24.501")
    chunk_b = _make_chunk("b", "text b", "24.501")
    chunk_c = _make_chunk("c", "text c", "24.501")

    dense_results = [RetrievedChunk(chunk=chunk_a, score=0.9), RetrievedChunk(chunk=chunk_b, score=0.5)]
    sparse_results = [RetrievedChunk(chunk=chunk_c, score=0.9), RetrievedChunk(chunk=chunk_a, score=0.4)]

    fused = reciprocal_rank_fusion([dense_results, sparse_results])

    # chunk_a appears rank-1 in dense and rank-2 in sparse -> highest fused score.
    assert fused[0].chunk.chunk_id == "a"


def test_reciprocal_rank_fusion_deduplicates_by_chunk_id() -> None:
    chunk_a = _make_chunk("a", "text a", "24.501")
    dense_results = [RetrievedChunk(chunk=chunk_a, score=0.9)]
    sparse_results = [RetrievedChunk(chunk=chunk_a, score=0.8)]

    fused = reciprocal_rank_fusion([dense_results, sparse_results])

    assert len(fused) == 1


def test_full_retriever_pipeline_end_to_end() -> None:
    client, provider = _seeded_client(COLLECTION + "_e2e")
    reranker = FakeReranker()
    retriever = Retriever(
        client=client,
        collection_name=COLLECTION + "_e2e",
        embedding_provider=provider,
        reranker=reranker,
        config=RetrievalConfig(dense_top_k=10, sparse_top_k=10, rerank_top_k=3),
    )

    results = retriever.retrieve("registration T3510 timer", release="Rel-18")

    assert len(results) <= 3
    assert all(r.chunk.release == "Rel-18" for r in results)


def test_fake_reranker_truncates_to_top_k() -> None:
    reranker = FakeReranker()
    chunks = [RetrievedChunk(chunk=_make_chunk(f"c{i}", f"registration text {i}", "24.501"), score=0.0) for i in range(5)]

    result = reranker.rerank("registration", chunks, top_k=2)

    assert len(result) == 2


def _build_retriever(collection: str, client, provider, reranker) -> Retriever:
    return Retriever(
        client=client,
        collection_name=collection,
        embedding_provider=provider,
        reranker=reranker,
        config=RetrievalConfig(dense_top_k=10, sparse_top_k=10, rerank_top_k=10),
    )


def test_retriever_auto_scopes_to_query_derived_spec_number() -> None:
    # Query mentions 23.501 (no explicit spec_number arg) -> only 23.501 chunks.
    client, provider = _seeded_client(COLLECTION + "_auto_spec")
    retriever = _build_retriever(COLLECTION + "_auto_spec", client, provider, FakeReranker())

    results = retriever.retrieve("What does AMF do in 23.501?", release="Rel-18")
    assert results  # something was retrieved
    assert all(r.chunk.spec_number == "23.501" for r in results)
    assert all(r.chunk.release == "Rel-18" for r in results)


def test_retriever_explicit_spec_number_overrides_query() -> None:
    # Explicit arg (24.501) must win even though the query text says 23.501.
    client, provider = _seeded_client(COLLECTION + "_explicit_wins")
    retriever = _build_retriever(COLLECTION + "_explicit_wins", client, provider, FakeReranker())

    results = retriever.retrieve(
        "What does AMF do in 23.501?", release="Rel-18", spec_number="24.501"
    )
    assert results
    assert all(r.chunk.spec_number == "24.501" for r in results)


def test_retriever_no_spec_filter_when_query_has_no_spec() -> None:
    # Generic query, no spec mentioned -> spans the whole release corpus.
    client, provider = _seeded_client(COLLECTION + "_no_spec")
    retriever = _build_retriever(COLLECTION + "_no_spec", client, provider, FakeReranker())

    results = retriever.retrieve("What is the role of the AMF?", release="Rel-18")
    # c1/c2 (24.501) and c3 (23.501) may both appear; c4 (Rel-17) must not.
    assert all(r.chunk.release == "Rel-18" for r in results)
    assert not any(r.chunk.chunk_id == "c4" for r in results)
    # No single spec_number is forced across the whole result set.
    spec_numbers = {r.chunk.spec_number for r in results}
    assert len(spec_numbers) > 1


def test_retriever_release_never_overridden_by_query() -> None:
    # Query including a spec number must still not leak Rel-17 content.
    client, provider = _seeded_client(COLLECTION + "_release_safe")
    retriever = _build_retriever(COLLECTION + "_release_safe", client, provider, FakeReranker())

    results = retriever.retrieve("Tell me about 24.501", release="Rel-18")
    assert all(r.chunk.release == "Rel-18" for r in results)
    assert not any(r.chunk.chunk_id == "c4" for r in results)
