from __future__ import annotations

from fastapi.testclient import TestClient

from app.dependencies import get_generator, get_retriever
from app.generation.evidence_gate import EvidenceGate
from app.generation.generator import GroundedGenerator
from app.generation.llm import FakeLLMProvider
from app.main import app
from app.models.schema import Chunk, ContentType
from app.retrieval.embeddings import FakeEmbeddingProvider
from app.retrieval.qdrant_store import ensure_collection, get_client, upsert_chunks
from app.retrieval.reranker import FakeReranker
from app.retrieval.retriever import RetrievalConfig, Retriever


def _chunk(
    chunk_id: str,
    spec_number: str,
    series: str,
    clause_number: str,
    clause_title: str,
    text: str,
    hash_char: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        spec_number=spec_number,
        series=series,
        release="Rel-18",
        release_number=18,
        version="18.9.0",
        title="Test spec",
        clause_number=clause_number,
        clause_title=clause_title,
        clause_path=["5", "5.5", clause_number],
        content_type=ContentType.PARAGRAPH,
        text=text,
        token_count=10,
        chunk_index=0,
        source_file="test.docx",
        source_url="https://example.com",
        source_locator=f"TS {spec_number} v18.9.0, Clause {clause_number}",
        content_hash="sha256:" + hash_char * 64,
    )


def _seeded_retriever() -> Retriever:
    client = get_client(":memory:")
    collection = "api_test_collection"
    ensure_collection(client, collection, dense_dim=8)
    provider = FakeEmbeddingProvider(dense_dim=8)

    chunk = _chunk(
        chunk_id="c1",
        spec_number="24.501",
        series="24",
        clause_number="5.5.1",
        clause_title="Registration",
        text="The UE shall send a REGISTRATION REQUEST message to the AMF.",
        hash_char="a",
    )
    upsert_chunks(client, collection, [chunk], provider)

    return Retriever(
        client=client,
        collection_name=collection,
        embedding_provider=provider,
        reranker=FakeReranker(),
        config=RetrievalConfig(dense_top_k=10, sparse_top_k=10, rerank_top_k=5),
    )


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_chat_endpoint_returns_grounded_answer() -> None:
    """End-to-end: /chat retrieves, passes the evidence gate, generates an
    answer with a valid citation, and returns the source document."""
    retriever = _seeded_retriever()
    generator = GroundedGenerator(
        llm_provider=FakeLLMProvider(response="The UE sends a REGISTRATION REQUEST [E1]."),
        evidence_gate=EvidenceGate(score_threshold=0.0),
    )
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: generator
    client = TestClient(app)

    response = client.post("/chat", json={"query": "What does the UE send?", "release": "Rel-18"})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert "[E1]" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["spec_number"] == "24.501"
