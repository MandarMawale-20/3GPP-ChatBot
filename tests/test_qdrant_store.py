from __future__ import annotations

from app.models.schema import Chunk, ContentType
from app.retrieval.embeddings import FakeEmbeddingProvider
from app.retrieval.qdrant_store import (
    chunk_point_id,
    ensure_collection,
    get_client,
    retrieve_existing_hashes,
    upsert_chunks,
)

COLLECTION = "3gpp_standards_test"


def _make_chunk(chunk_id: str, spec_number: str = "24.501", clause: str = "5.5.1", release: str = "Rel-18") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        spec_number=spec_number,
        series="24",
        release=release,
        release_number=18,
        version="18.9.0",
        title="Test spec",
        clause_number=clause,
        clause_title="Registration procedure",
        clause_path=["5", "5.5", clause],
        content_type=ContentType.PARAGRAPH,
        text="The UE shall send a REGISTRATION REQUEST message to the AMF.",
        token_count=10,
        chunk_index=0,
        source_file="24501-i90.docx",
        source_url="https://example.com",
        source_locator="TS 24.501 v18.9.0, Clause 5.5.1",
        content_hash="sha256:" + "a" * 64,
    )


def test_chunk_point_id_is_deterministic() -> None:
    id_a = chunk_point_id("24.501_R18_18.9.0_5.5.1_000")
    id_b = chunk_point_id("24.501_R18_18.9.0_5.5.1_000")
    id_c = chunk_point_id("24.501_R18_18.9.0_5.5.1_001")

    assert id_a == id_b
    assert id_a != id_c


def test_ensure_collection_is_idempotent() -> None:
    client = get_client(":memory:")
    ensure_collection(client, COLLECTION, dense_dim=8)
    ensure_collection(client, COLLECTION, dense_dim=8)  # must not raise on second call

    assert client.collection_exists(COLLECTION)


def test_upsert_and_retrieve_chunk_roundtrip() -> None:
    client = get_client(":memory:")
    ensure_collection(client, COLLECTION + "_roundtrip", dense_dim=8)
    provider = FakeEmbeddingProvider(dense_dim=8)
    chunk = _make_chunk("24.501_R18_18.9.0_5.5.1_000")

    count = upsert_chunks(client, COLLECTION + "_roundtrip", [chunk], provider, batch_size=16)
    assert count == 1

    point_id = chunk_point_id(chunk.chunk_id)
    retrieved = client.retrieve(collection_name=COLLECTION + "_roundtrip", ids=[point_id], with_payload=True)
    assert len(retrieved) == 1
    retrieved_chunk = Chunk.model_validate(retrieved[0].payload)
    assert retrieved_chunk.chunk_id == chunk.chunk_id
    assert retrieved_chunk.text == chunk.text


def test_upsert_is_idempotent_for_same_chunk_id() -> None:
    client = get_client(":memory:")
    collection = COLLECTION + "_idempotent"
    ensure_collection(client, collection, dense_dim=8)
    provider = FakeEmbeddingProvider(dense_dim=8)
    chunk = _make_chunk("24.501_R18_18.9.0_5.5.1_000")

    upsert_chunks(client, collection, [chunk], provider)
    upsert_chunks(client, collection, [chunk], provider)  # re-ingest same chunk_id

    count_result = client.count(collection_name=collection)
    assert count_result.count == 1  # not duplicated


def test_retrieve_existing_hashes_maps_chunk_id_to_hash() -> None:
    client = get_client(":memory:")
    collection = COLLECTION + "_hashes"
    ensure_collection(client, collection, dense_dim=8)
    provider = FakeEmbeddingProvider(dense_dim=8)
    chunk = _make_chunk("24.501_R18_18.9.0_5.5.1_000")

    upsert_chunks(client, collection, [chunk], provider, skip_unchanged=False)
    hashes = retrieve_existing_hashes(client, collection)

    assert hashes == {"24.501_R18_18.9.0_5.5.1_000": chunk.content_hash}


def test_retrieve_existing_hashes_empty_for_missing_collection() -> None:
    client = get_client(":memory:")
    assert retrieve_existing_hashes(client, COLLECTION + "_does_not_exist") == {}


def test_upsert_skips_unchanged_chunks_on_reingest() -> None:
    client = get_client(":memory:")
    collection = COLLECTION + "_skip"
    ensure_collection(client, collection, dense_dim=8)
    provider = FakeEmbeddingProvider(dense_dim=8)

    chunk_a = _make_chunk("24.501_R18_18.9.0_5.5.1_000", clause="5.5.1")
    chunk_b = _make_chunk("24.501_R18_18.9.0_5.5.1_001", clause="5.5.2")

    upsert_chunks(client, collection, [chunk_a, chunk_b], provider)

    # Re-ingest both (unchanged) plus one brand-new chunk and one edited chunk.
    chunk_b_edited = _make_chunk("24.501_R18_18.9.0_5.5.1_001", clause="5.5.2")
    chunk_b_edited.content_hash = "sha256:" + "e" * 64  # content changed
    chunk_c = _make_chunk("24.501_R18_18.9.0_5.5.1_002", clause="5.5.3")

    count = upsert_chunks(client, collection, [chunk_a, chunk_b_edited, chunk_c], provider)

    # Only the edited chunk (b) and the new chunk (c) are re-embedded/upserted.
    assert count == 2

    # Final collection still holds exactly 3 points (a, b-edited, c).
    assert client.count(collection_name=collection).count == 3


def test_upsert_reembeds_all_when_skip_disabled() -> None:
    client = get_client(":memory:")
    collection = COLLECTION + "_no_skip"
    ensure_collection(client, collection, dense_dim=8)
    provider = FakeEmbeddingProvider(dense_dim=8)
    chunk = _make_chunk("24.501_R18_18.9.0_5.5.1_000")

    upsert_chunks(client, collection, [chunk], provider, skip_unchanged=False)
    count = upsert_chunks(client, collection, [chunk], provider, skip_unchanged=False)

    assert count == 1  # re-embedded even though unchanged
