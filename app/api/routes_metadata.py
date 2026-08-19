"""`GET /metadata/documents` — list the corpus allowlist for the UI.

The frontend uses this to populate its document selector. The list derives
from `configs/corpus.yaml` (the frozen Rel-18 allowlist), never from the
Qdrant collection contents — the allowlist is the single source of truth
for *what* this deployment may index/query.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import DocumentInfo, DocumentsResponse
from app.config import get_settings

router = APIRouter()


@router.get("/metadata/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    config = get_settings()
    docs = [
        DocumentInfo(
            spec_number=d["spec_number"],
            title=d["title"],
            series=d["series"],
            release=d["release"],
        )
        for d in config.allowed_documents()  # all enabled releases
    ]
    return DocumentsResponse(release=config.default_release, documents=docs)