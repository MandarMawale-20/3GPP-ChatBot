"""`POST /search` — transparent retrieval debugging endpoint (SRD §42)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.schemas import SearchRequest, SearchResponse, SearchResultResponse
from app.dependencies import get_retriever
from app.retrieval.retriever import Retriever

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, retriever: Retriever = Depends(get_retriever)) -> SearchResponse:
    # Debug endpoint: bypasses the evidence gate/LLM entirely so the raw
    # hybrid+reranked retrieval output can be inspected directly.
    retrieved = retriever.retrieve(
        request.query, request.release, request.spec_number, rerank_top_k=request.top_k
    )

    results = [
        SearchResultResponse(
            chunk_id=r.chunk.chunk_id,
            score=r.score,
            spec=r.chunk.spec_number,
            clause=r.chunk.clause_number,
            content_type=r.chunk.content_type.value if hasattr(r.chunk.content_type, "value") else r.chunk.content_type,
            text=r.chunk.text,
        )
        for r in retrieved
    ]
    return SearchResponse(results=results)
