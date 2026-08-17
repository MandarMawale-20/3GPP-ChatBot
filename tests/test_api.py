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


def _seeded_retriever_two_specs() -> Retriever:
    """Retriever seeded with chunks from two different specs, to verify
    that spec_number filtering actually restricts retrieval."""
    client = get_client(":memory:")
    collection = "api_test_collection_two_specs"
    ensure_collection(client, collection, dense_dim=8)
    provider = FakeEmbeddingProvider(dense_dim=8)

    chunks = [
        _chunk(
            chunk_id="c1",
            spec_number="24.501",
            series="24",
            clause_number="5.5.1",
            clause_title="Registration",
            text="The UE shall send a REGISTRATION REQUEST message to the AMF.",
            hash_char="a",
        ),
        _chunk(
            chunk_id="c2",
            spec_number="23.501",
            series="23",
            clause_number="4.2",
            clause_title="Session continuity",
            text="The PDU session is established by the SMF.",
            hash_char="b",
        ),
    ]
    upsert_chunks(client, collection, chunks, provider)

    return Retriever(
        client=client,
        collection_name=collection,
        embedding_provider=provider,
        reranker=FakeReranker(),
        config=RetrievalConfig(dense_top_k=10, sparse_top_k=10, rerank_top_k=5),
    )


def _make_client(llm_response: str, score_threshold: float = 0.0) -> TestClient:
    retriever = _seeded_retriever()
    generator = GroundedGenerator(
        llm_provider=FakeLLMProvider(response=llm_response),
        evidence_gate=EvidenceGate(score_threshold=score_threshold),
    )

    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: generator
    return TestClient(app)


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_health_endpoint_reports_status() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_search_endpoint_returns_results() -> None:
    client = _make_client(llm_response="unused")
    response = client.post("/search", json={"query": "registration", "release": "Rel-18", "top_k": 5})

    assert response.status_code == 200
    body = response.json()
    assert "results" in body
    assert len(body["results"]) >= 1
    assert body["results"][0]["spec"] == "24.501"


def test_chat_endpoint_returns_grounded_answer() -> None:
    client = _make_client(llm_response="The UE sends a REGISTRATION REQUEST [E1].", score_threshold=0.0)
    response = client.post("/chat", json={"query": "What does the UE send?", "release": "Rel-18"})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert "[E1]" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["spec_number"] == "24.501"


def test_chat_endpoint_abstains_when_gate_threshold_too_high() -> None:
    client = _make_client(llm_response="Should not be used", score_threshold=999.0)
    response = client.post("/chat", json={"query": "What does the UE send?", "release": "Rel-18"})

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is True
    assert body["sources"] == []


def test_chat_endpoint_rejects_empty_query() -> None:
    client = _make_client(llm_response="unused")
    response = client.post("/chat", json={"query": "", "release": "Rel-18"})

    assert response.status_code == 422  # pydantic min_length validation


def test_metadata_documents_endpoint_lists_corpus() -> None:
    client = TestClient(app)
    response = client.get("/metadata/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["release"] == "Rel-18"
    assert len(body["documents"]) >= 8  # core + extended allowlist
    spec_numbers = {d["spec_number"] for d in body["documents"]}
    assert "23.501" in spec_numbers
    assert "24.501" in spec_numbers
    assert all({"spec_number", "title", "series"} <= set(d) for d in body["documents"])


def test_chat_omitting_spec_number_searches_all_documents() -> None:
    """'All Documents' (spec_number omitted) must return results across
    every indexed spec, not just one."""
    retriever = _seeded_retriever_two_specs()
    generator = GroundedGenerator(
        llm_provider=FakeLLMProvider(response="The UE sends a REGISTRATION REQUEST [E1]."),
        evidence_gate=EvidenceGate(score_threshold=0.0),
    )
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: generator
    client = TestClient(app)

    response = client.post("/chat", json={"query": "What does the UE send?", "release": "Rel-18"})

    assert response.status_code == 200
    # The fake embedder returns the same vector for everything, so both
    # specs' chunks should be retrieved and passed to the generator.
    body = response.json()
    assert body["abstained"] is False
    assert "[E1]" in body["answer"]


def test_chat_with_spec_number_restricts_retrieval() -> None:
    """A specific document selection must restrict retrieval to that spec."""
    retriever = _seeded_retriever_two_specs()
    generator = GroundedGenerator(
        llm_provider=FakeLLMProvider(response="The UE sends a REGISTRATION REQUEST [E1]."),
        evidence_gate=EvidenceGate(score_threshold=0.0),
    )
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_generator] = lambda: generator
    client = TestClient(app)

    response = client.post(
        "/chat",
        json={"query": "What does the UE send?", "release": "Rel-18", "spec_number": "24.501"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["abstained"] is False
    assert "[E1]" in body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["spec_number"] == "24.501"


def test_search_with_spec_number_restricts_results() -> None:
    retriever = _seeded_retriever_two_specs()
    app.dependency_overrides[get_retriever] = lambda: retriever
    client = TestClient(app)

    response = client.post(
        "/search",
        json={"query": "session", "release": "Rel-18", "spec_number": "23.501", "top_k": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) >= 1
    # All returned chunks must belong to the selected spec.
    assert all(r["spec"] == "23.501" for r in body["results"])
