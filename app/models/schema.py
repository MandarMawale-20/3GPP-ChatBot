"""Canonical data models for the ingestion -> retrieval -> generation pipeline.

`Chunk` is the canonical JSONL record, the Qdrant payload shape, and the
object citations are generated from. Every downstream component consumes
this exact schema, so a change here is a breaking change everywhere.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

_SHA256_HEX_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ContentType(str, Enum):
    """Structural unit a chunk represents.

    An explicit enum (not free text) so retrieval filtering and evaluation
    slicing can rely on a fixed set of values.
    """

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    PROCEDURE = "procedure"
    TABLE = "table"
    ASN1 = "asn1"
    FIGURE = "figure"
    ANNEX = "annex"
    NOTE = "note"


class TableCell(BaseModel):
    """A single (possibly merged) table cell.

    `rowspan`/`colspan` capture OpenXML merge geometry so the table parser
    can normalize inherited values instead of emitting empty continuation
    cells.
    """

    text: str
    rowspan: int = Field(default=1, ge=1)
    colspan: int = Field(default=1, ge=1)
    is_merged_continuation: bool = Field(
        default=False,
        description="True if this cell's value was inherited from a merge, "
        "not present verbatim in the source XML.",
    )


class TableData(BaseModel):
    """Structured table representation, kept alongside the Markdown form.

    Both a structured representation (for programmatic use) and a
    retrieval-friendly Markdown form (stored in `Chunk.text`) are kept;
    this model is the structured half.
    """

    headers: list[str] = Field(default_factory=list)
    rows: list[list[TableCell]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rows_match_header_width(self) -> TableData:
        # A malformed merge-normalization pass can silently drop or
        # duplicate cells; catch width mismatches at construction time.
        if self.headers:
            expected = len(self.headers)
            for i, row in enumerate(self.rows):
                width = sum(c.colspan for c in row)
                if width != expected:
                    raise ValueError(
                        f"Table row {i} has effective width {width}, "
                        f"expected {expected} (from headers)"
                    )
        return self


class SourceDocument(BaseModel):
    """Provenance/integrity record for one downloaded raw source file.

    Persisted as JSON alongside the raw file and referenced by every chunk
    derived from it via `source_file` / `content_hash`.
    """

    spec_number: str
    release: str
    version: str
    filename: str
    sha256: str
    downloaded_at: str  # ISO-8601 timestamp; kept as str to stay JSON-simple
    source_url: str

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", v.lower()):
            raise ValueError(f"sha256 must be a 64-char hex digest, got: {v!r}")
        return v.lower()

    @field_validator("release")
    @classmethod
    def _validate_release_format(cls, v: str) -> str:
        if not re.fullmatch(r"Rel-\d+", v):
            raise ValueError(f"release must look like 'Rel-18', got: {v!r}")
        return v


class Chunk(BaseModel):
    """The canonical retrieval unit. One instance == one JSONL line == one
    Qdrant point payload.
    """

    # Identity
    chunk_id: str = Field(
        description="e.g. '24.501_R18_18.9.0_5.5.1_003'. Must be globally "
        "unique within the collection."
    )

    # Document identity
    spec_number: str = Field(description="e.g. '24.501'")
    document_type: str = Field(default="TS", description="'TS' or 'TR'")
    series: str = Field(description="e.g. '24' (the series a spec belongs to)")
    release: str = Field(description="e.g. 'Rel-18'")
    release_number: int = Field(description="e.g. 18")
    version: str = Field(description="e.g. '18.9.0'")
    title: str

    # Structural location
    clause_number: str = Field(default="", description="e.g. '5.5.1'")
    clause_title: str = Field(default="")
    clause_path: list[str] = Field(
        default_factory=list,
        description="e.g. ['5', '5.5', '5.5.1'] — full ancestry, root first.",
    )
    parent_clause: str = Field(default="")
    parent_title: str = Field(default="")

    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    # Content
    content_type: ContentType
    text: str = Field(description="Retrieval-friendly source text (Markdown "
        "for tables). This is what gets embedded and shown as evidence — "
        "distinct from any richer `embedding_text` built at embed time.")
    table_data: TableData | None = Field(
        default=None,
        description="Populated only when content_type == TABLE.",
    )
    token_count: int = Field(ge=0)
    chunk_index: int = Field(ge=0, description="Position among siblings under the same clause.")

    # Parent/child linkage
    parent_chunk_id: str | None = Field(
        default=None,
        description="Set on child chunks; None on standalone/parent chunks.",
    )

    # Flags used for Qdrant payload filtering
    is_normative: bool = Field(default=True)
    is_annex: bool = Field(default=False)
    is_table: bool = Field(default=False)
    is_figure: bool = Field(default=False)
    is_asn1: bool = Field(default=False)

    # Provenance (citation generation depends on these)
    source_file: str
    source_url: str
    source_locator: str = Field(
        description="Human-readable citation string, e.g. "
        "'TS 24.501 v18.9.0, Clause 5.5.1, Pages 123-125'."
    )
    content_hash: str = Field(description="'sha256:<64 hex chars>' of `text`.")

    @field_validator("content_hash")
    @classmethod
    def _validate_content_hash(cls, v: str) -> str:
        if not _SHA256_HEX_RE.match(v):
            raise ValueError(
                f"content_hash must match 'sha256:<64 hex chars>', got: {v!r}"
            )
        return v

    @field_validator("release")
    @classmethod
    def _validate_release_format(cls, v: str) -> str:
        if not re.fullmatch(r"Rel-\d+", v):
            raise ValueError(f"release must look like 'Rel-18', got: {v!r}")
        return v

    @model_validator(mode="after")
    def _flags_consistent_with_content_type(self) -> Chunk:
        # These booleans exist as fast Qdrant payload-index filters. If
        # they drift from content_type, filtered retrieval silently returns
        # wrong results, so enforce consistency at construction time.
        expected = {
            ContentType.TABLE: "is_table",
            ContentType.FIGURE: "is_figure",
            ContentType.ASN1: "is_asn1",
            ContentType.ANNEX: "is_annex",
        }
        for content_type, flag_name in expected.items():
            if self.content_type == content_type and not getattr(self, flag_name):
                raise ValueError(
                    f"content_type={content_type.value!r} requires {flag_name}=True"
                )
        if self.content_type == ContentType.TABLE and self.table_data is None:
            raise ValueError("content_type=TABLE requires table_data to be set")
        return self

    model_config = {
        "use_enum_values": False,
        "json_schema_extra": {
            "example": {
                "chunk_id": "24.501_R18_18.9.0_5.5.1_003",
                "spec_number": "24.501",
                "document_type": "TS",
                "series": "24",
                "release": "Rel-18",
                "release_number": 18,
                "version": "18.9.0",
                "title": "Non-Access-Stratum (NAS) protocol for 5G System (5GS)",
                "clause_number": "5.5.1",
                "clause_title": "Registration procedure",
                "clause_path": ["5", "5.5", "5.5.1"],
                "parent_clause": "5.5",
                "parent_title": "Registration procedure",
                "page_start": 123,
                "page_end": 125,
                "content_type": "procedure",
                "text": "The UE shall send a REGISTRATION REQUEST message...",
                "token_count": 486,
                "chunk_index": 3,
                "parent_chunk_id": "24.501_R18_18.9.0_5.5.1_PARENT",
                "is_normative": True,
                "is_annex": False,
                "is_table": False,
                "is_figure": False,
                "is_asn1": False,
                "source_file": "24501-i90.docx",
                "source_url": "https://www.3gpp.org/ftp/specs/latest/Rel-18/24_series/",
                "source_locator": "TS 24.501 v18.9.0, Clause 5.5.1, Pages 123-125",
                "content_hash": "sha256:" + "0" * 64,
            }
        },
    }
