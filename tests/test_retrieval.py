from __future__ import annotations

from app.models.schema import Chunk, ContentType
from app.retrieval.dense import RetrievedChunk, dense_search
from app.retrieval.embeddings import FakeEmbeddingProvider
from app.retrieval.hybrid import build_release_spec_filter
from app.retrieval.qdrant_store import ensure_collection, get_client, upsert_chunks
from app.retrieval.reranker import FakeReranker
from app.retrieval.retriever import RetrievalConfig, Retriever

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


def _build_retriever(collection: str, client, provider, reranker) -> Retriever:
    return Retriever(
        client=client,
        collection_name=collection,
        embedding_provider=provider,
        reranker=reranker,
        config=RetrievalConfig(dense_top_k=10, sparse_top_k=10, rerank_top_k=10),
    )


def test_metadata_filter_isolates_spec_number() -> None:
    """A Qdrant payload filter scopes dense retrieval to one spec number."""
    client, provider = _seeded_client(COLLECTION + "_spec_filter")
    query_filter = build_release_spec_filter("Rel-18", spec_number="23.501")

    query_vector = provider.embed_dense(["registration"])[0]
    results = dense_search(client, COLLECTION + "_spec_filter", query_vector, top_k=10, query_filter=query_filter)

    assert all(r.chunk.spec_number == "23.501" for r in results)


def test_retriever_release_never_overridden_by_query() -> None:
    """A query naming a spec must not leak content from another release."""
    client, provider = _seeded_client(COLLECTION + "_release_safe")
    retriever = _build_retriever(COLLECTION + "_release_safe", client, provider, FakeReranker())

    results = retriever.retrieve("Tell me about 24.501", release="Rel-18")
    assert all(r.chunk.release == "Rel-18" for r in results)
    assert not any(r.chunk.chunk_id == "c4" for r in results)


def test_retriever_all_releases_mode() -> None:
    """When release is None, retrieval spans all indexed releases."""
    client, provider = _seeded_client(COLLECTION + "_all_releases")
    retriever = _build_retriever(COLLECTION + "_all_releases", client, provider, FakeReranker())

    results = retriever.retrieve("Tell me about 24.501", release=None)
    # Should include both Rel-18 and Rel-17 content — no release filter applied.
    releases = {r.chunk.release for r in results}
    assert "Rel-17" in releases
    assert "Rel-18" in releases


def test_build_release_spec_filter_omits_release_when_none() -> None:
    """build_release_spec_filter with release=None produces no release condition."""
    from app.retrieval.hybrid import build_release_spec_filter

    f = build_release_spec_filter(None)
    # No must-conditions when release is None and spec_number is None.
    assert len(f.must) == 0

    f = build_release_spec_filter(None, spec_number="24.501")
    # Only the spec_number condition, no release condition.
    assert len(f.must) == 1
    assert f.must[0].key == "spec_number"
