"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api import routes_chat, routes_health, routes_metadata, routes_search
from app.logging_config import configure_logging

configure_logging()

app = FastAPI(
    title="3GPP Standards RAG Chatbot",
    description=(
        "Release-aware, clause-aware RAG API over official 3GPP "
        "technical specifications. Supports release-aware querying "
        "(specific release, specific spec, or all indexed releases). "
        "Answers are grounded in retrieved evidence and abstain when "
        "the corpus lacks sufficient support."
    ),
    version="0.1.0",
)

app.include_router(routes_health.router, tags=["health"])
app.include_router(routes_search.router, tags=["search"])
app.include_router(routes_chat.router, tags=["chat"])
app.include_router(routes_metadata.router, tags=["metadata"])
