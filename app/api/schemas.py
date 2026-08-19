"""API request/response models (SRD §40-§42)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    release: str | None = Field(default="Rel-18")
    spec_number: str | None = None


class SourceResponse(BaseModel):
    spec_number: str
    release: str
    version: str
    clause: str
    source_locator: str


class ChatResponse(BaseModel):
    answer: str
    abstained: bool
    confidence: float
    sources: list[SourceResponse]
    abstain_reason: str = ""


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    release: str | None = Field(default="Rel-18")
    spec_number: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class SearchResultResponse(BaseModel):
    chunk_id: str
    score: float
    spec: str
    clause: str
    content_type: str
    text: str


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]


class DocumentInfo(BaseModel):
    """A single allowlisted document in the corpus allowlist.

    `release` is always populated: a spec number (e.g. 24.501) may appear in
    more than one indexed release, so the release disambiguates which document
    the frontend selector is referring to.
    """

    spec_number: str
    title: str
    series: str
    release: str


class DocumentsResponse(BaseModel):
    release: str
    documents: list[DocumentInfo]


class HealthResponse(BaseModel):
    status: str
    qdrant_reachable: bool
    collection_exists: bool
