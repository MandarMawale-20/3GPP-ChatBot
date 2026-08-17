"""`GET /health` (SRD §41)."""

from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

from app.api.schemas import HealthResponse
from app.config import get_settings
from app.retrieval.qdrant_store import get_client

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings().settings
    qdrant_reachable = False
    collection_exists = False

    try:
        client = get_client(settings.qdrant_url, settings.qdrant_api_key)
        collection_exists = client.collection_exists(settings.qdrant_collection)
        qdrant_reachable = True
    except Exception as exc:  # noqa: BLE001 — health check must never raise, only report
        logger.warning("Qdrant health check failed: {}", exc)

    status = "ok" if qdrant_reachable and collection_exists else "degraded"
    return HealthResponse(status=status, qdrant_reachable=qdrant_reachable, collection_exists=collection_exists)
