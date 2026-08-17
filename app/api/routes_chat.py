"""`POST /chat` — retrieval + grounded generation in one call."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from loguru import logger

from app.api.schemas import ChatRequest, ChatResponse, SourceResponse
from app.dependencies import get_generator, get_retriever
from app.generation.generator import GroundedGenerator
from app.retrieval.retriever import Retriever

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    generator: GroundedGenerator = Depends(get_generator),
) -> ChatResponse:
    logger.info("Chat query: {!r} (release={})", request.query, request.release)

    retrieved = retriever.retrieve(request.query, request.release, request.spec_number)
    result = generator.answer(request.query, retrieved)

    sources = [
        SourceResponse(
            spec_number=c.spec_number,
            release=c.release,
            version=c.version,
            clause=c.clause_number,
            source_locator=c.source_locator,
        )
        for c in result.citations
    ]

    return ChatResponse(
        answer=result.answer,
        abstained=result.abstained,
        confidence=result.confidence,
        sources=sources,
        abstain_reason=result.abstain_reason,
    )
