"""End-to-end ingestion pipeline: download -> validate -> extract -> parse
-> chunk -> JSONL.

Two entry points:

- `process_local_document()` — parsing/chunking for an already-present DOCX
  file (used by tests and for manually downloaded archives).
- `download_and_process()` — full remote pipeline: resolve latest archive,
  download, validate, extract, convert if needed, then delegate to
  `process_local_document()`.

The network-dependent download step is decoupled from the testable parsing
step.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from app.config import get_settings
from app.models.schema import Chunk, SourceDocument
from ingestion.archive import find_document_file, safe_extract
from ingestion.chunker import DocumentMetadata, build_chunks
from ingestion.doc_converter import convert_doc_to_docx
from ingestion.docx_parser import parse_docx
from ingestion.downloader import (
    download_archive,
    fetch_directory_listing,
    find_latest_archive,
    parse_archive_links,
)
from ingestion.structure_parser import extract_structural_elements
from ingestion.validator import ExpectedDocument, validate_document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


def process_local_document(
    docx_path: Path,
    spec_number: str,
    series: str,
    release: str,
    release_number: int,
    version: str,
    title: str,
    source_file: str,
    source_url: str,
) -> list[Chunk]:
    """Parse and chunk a single already-downloaded DOCX file."""
    raw_elements = parse_docx(docx_path)
    structural_elements = extract_structural_elements(raw_elements)

    table_count = sum(1 for e in structural_elements if e.is_table)
    logger.info("Extracted {} tables", table_count)

    doc_meta = DocumentMetadata(
        spec_number=spec_number,
        document_type="TS",
        series=series,
        release=release,
        release_number=release_number,
        version=version,
        title=title,
        source_file=source_file,
        source_url=source_url,
    )
    return build_chunks(structural_elements, doc_meta)


def write_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json())
            f.write("\n")
    logger.info("Wrote {} chunks -> {}", len(chunks), output_path)


def read_jsonl(input_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks


def _find_title_in_corpus(spec_number: str, release: str) -> str:
    """Look up the allowlisted title for a spec within one release.

    Falls back to the bare spec number (never a fabricated title) if the spec
    isn't allowlisted — the caller is still responsible for validating the
    spec against the official directory before indexing.
    """
    cfg = get_settings().release_config(release)
    for group in cfg.get("documents", {}).values():
        for doc in group:
            if doc["spec_number"] == spec_number:
                return doc["title"]
    return spec_number


def _resolve_release(release: str | None) -> tuple[str, dict]:
    """Resolve a release name to (release, release_config), defaulting to the
    deployment default when unspecified."""
    if release is None:
        release = get_settings().default_release
    cfg = get_settings().release_config(release)
    return release, cfg


def download_and_process(
    spec_number: str,
    series: str,
    release: str | None = None,
    title: str | None = None,
) -> list[Chunk]:
    """Full remote pipeline for a single spec, driven by `configs/corpus.yaml`.

    `release` selects which release's allowlist + repository root to use; it
    defaults to the deployment `default_release` (env TARGET_RELEASE). Each
    release is namespaced on disk under its own subdirectory of `data/raw` and
    `data/processed` so the same spec number can coexist across releases
    (e.g. 24.501 in both Rel-17 and Rel-18) without file/ provenance
    collisions.

    Requires outbound network access to the official 3GPP repository. Call
    `process_local_document()` directly if the DOCX file is already present.
    """
    release, release_cfg = _resolve_release(release)
    release_number: int = release_cfg["release_number"]
    repo_root: str = release_cfg["sources"]["repository_root"]
    if title is None:
        title = _find_title_in_corpus(spec_number, release)

    series_url = f"{repo_root.rstrip('/')}/{series}_series/"
    html = fetch_directory_listing(series_url)
    archives = parse_archive_links(html, series_url)
    archive = find_latest_archive(archives, spec_number, release_number)

    zip_path = RAW_DIR / release.lower() / spec_number / archive.filename
    download_archive(archive, zip_path)

    expected = ExpectedDocument(spec_number=spec_number, release=release, release_number=release_number)
    validated = validate_document(archive.filename, zip_path, expected)

    extract_dir = zip_path.parent / "extracted"
    if zip_path.suffix.lower() == ".zip":
        safe_extract(zip_path, extract_dir)
        doc_path = find_document_file(extract_dir)
    else:
        doc_path = zip_path  # some smaller specs are published as a bare DOCX

    if doc_path.suffix.lower() == ".doc":
        doc_path = convert_doc_to_docx(doc_path, extract_dir)

    # Persist provenance metadata alongside the raw file; citation
    # generation and audit trails trace back to this. Namespaced by release so
    # a spec present in multiple releases keeps separate provenance records.
    source_doc = SourceDocument(
        spec_number=validated.spec_number,
        release=validated.release,
        version=validated.version,
        filename=validated.filename,
        sha256=validated.sha256,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
        source_url=archive.url,
    )
    metadata_path = METADATA_DIR / release.lower() / f"{spec_number}.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(source_doc.model_dump_json(indent=2), encoding="utf-8")

    chunks = process_local_document(
        docx_path=doc_path,
        spec_number=validated.spec_number,
        series=series,
        release=validated.release,
        release_number=validated.release_number,
        version=validated.version,
        title=title,
        source_file=validated.filename,
        source_url=archive.url,
    )

    # Namespaced by release: the same spec_number resolves to distinct files
    # per release (Rel-17 24.501 vs Rel-18 24.501).
    write_jsonl(chunks, PROCESSED_DIR / release.lower() / f"{spec_number}.jsonl")
    return chunks