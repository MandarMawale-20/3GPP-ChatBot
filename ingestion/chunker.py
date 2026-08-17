"""Clause-aware chunking.

Per clause:
1. TABLE and ASN1 elements are isolated into their own chunk (atomic units).
2. Other elements are packed greedily into blocks targeting
   `target_tokens_max` tokens.
3. If a clause produces more than one block, a synthetic PARENT chunk with
   the clause's full text is emitted and referenced via `parent_chunk_id`,
   so retrieval can expand a matched child to full clause context.
4. Overlap is only applied between two non-atomic adjacent blocks, never
   across a table/ASN.1 boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import groupby

import tiktoken
from loguru import logger

from app.config import get_settings
from app.models.schema import Chunk, ContentType
from ingestion.structure_parser import StructuralElement

# tiktoken downloads its vocabulary on first use, which can fail offline.
# Fall back to a ~4 chars/token heuristic so chunking still works; this
# only affects chunk sizing precision, not content correctness.
_encoding: tiktoken.Encoding | None = None
_encoding_load_attempted = False


def _get_encoding() -> tiktoken.Encoding | None:
    global _encoding, _encoding_load_attempted
    if not _encoding_load_attempted:
        _encoding_load_attempted = True
        try:
            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:  # noqa: BLE001 — any load failure falls back
            logger.warning(
                "tiktoken encoding unavailable ({}); falling back to a "
                "character-count token approximation.",
                exc,
            )
            _encoding = None
    return _encoding


@dataclass(frozen=True)
class DocumentMetadata:
    """Source document fields carried on every chunk."""

    spec_number: str
    document_type: str
    series: str
    release: str
    release_number: int
    version: str
    title: str
    source_file: str
    source_url: str


def count_tokens(text: str) -> int:
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text))
    if not text:
        return 0
    return max(1, len(text) // 4)


def _content_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dominant_content_type(elements: list[StructuralElement]) -> ContentType:
    types = {e.content_type for e in elements}
    if len(types) == 1:
        return types.pop()
    return ContentType.PARAGRAPH


@dataclass
class _Block:
    elements: list[StructuralElement]

    @property
    def text(self) -> str:
        return "\n\n".join(e.text for e in self.elements if e.text)

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)

    @property
    def content_type(self) -> ContentType:
        return _dominant_content_type(self.elements)

    @property
    def is_atomic(self) -> bool:
        return self.content_type in (ContentType.TABLE, ContentType.ASN1)


def _pack_clause_elements(elements: list[StructuralElement], max_tokens: int) -> list[_Block]:
    """Greedy packing with forced isolation of table/ASN.1 elements."""
    blocks: list[_Block] = []
    current: list[StructuralElement] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            blocks.append(_Block(elements=current))
        current = []
        current_tokens = 0

    for element in elements:
        if element.content_type in (ContentType.TABLE, ContentType.ASN1):
            flush()
            blocks.append(_Block(elements=[element]))
            continue

        element_tokens = count_tokens(element.text)
        if current and current_tokens + element_tokens > max_tokens:
            flush()

        current.append(element)
        current_tokens += element_tokens

    flush()
    return blocks


def _apply_overlap(blocks: list[_Block], overlap_tokens: int) -> list[str]:
    """Prepend a trailing slice of the previous block's text to each
    subsequent block, but only across two non-atomic blocks.
    """
    encoding = _get_encoding()
    texts: list[str] = []
    for i, block in enumerate(blocks):
        text = block.text
        if i > 0 and not block.is_atomic and not blocks[i - 1].is_atomic:
            prev_text = blocks[i - 1].text
            if encoding is not None:
                prev_tokens = encoding.encode(prev_text)
                overlap_slice = encoding.decode(prev_tokens[-overlap_tokens:])
            else:
                prev_words = prev_text.split()
                approx_word_count = max(1, int(overlap_tokens * 0.75))
                overlap_slice = " ".join(prev_words[-approx_word_count:])
            text = f"{overlap_slice}\n\n{text}"
        texts.append(text)
    return texts


def build_chunks(elements: list[StructuralElement], doc_meta: DocumentMetadata) -> list[Chunk]:
    """Build the ordered list of Chunk records for one parsed document."""
    settings = get_settings().tuning["chunking"]
    max_tokens = settings["target_tokens_max"]
    overlap_tokens = settings["overlap_tokens_min"]

    # HEADING elements only drive clause tracking; their title is already
    # captured as clause_title on every element beneath them.
    body_elements = [e for e in elements if e.content_type != ContentType.HEADING]

    chunks: list[Chunk] = []

    for clause_number, group_iter in groupby(body_elements, key=lambda e: e.clause_number):
        group = list(group_iter)
        if not group:
            continue

        blocks = _pack_clause_elements(group, max_tokens)
        if not blocks:
            continue

        texts = _apply_overlap(blocks, overlap_tokens)
        head = group[0]
        parent_chunk_id: str | None = None

        if len(blocks) > 1:
            parent_chunk_id = _chunk_id(doc_meta, clause_number, "PARENT")
            full_text = "\n\n".join(b.text for b in blocks)
            chunks.append(
                Chunk(
                    chunk_id=parent_chunk_id,
                    spec_number=doc_meta.spec_number,
                    document_type=doc_meta.document_type,
                    series=doc_meta.series,
                    release=doc_meta.release,
                    release_number=doc_meta.release_number,
                    version=doc_meta.version,
                    title=doc_meta.title,
                    clause_number=clause_number,
                    clause_title=head.clause_title,
                    clause_path=head.clause_path,
                    parent_clause=head.parent_clause,
                    parent_title=head.parent_title,
                    content_type=ContentType.PARAGRAPH,
                    text=full_text,
                    token_count=count_tokens(full_text),
                    chunk_index=0,
                    parent_chunk_id=None,
                    is_annex=head.is_annex,
                    is_table=False,
                    is_figure=any(e.is_figure for e in group),
                    is_asn1=False,
                    source_file=doc_meta.source_file,
                    source_url=doc_meta.source_url,
                    source_locator=_source_locator(doc_meta, clause_number),
                    content_hash=_content_hash(full_text),
                )
            )

        for index, (block, text) in enumerate(zip(blocks, texts)):
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(doc_meta, clause_number, f"{index:03d}"),
                    spec_number=doc_meta.spec_number,
                    document_type=doc_meta.document_type,
                    series=doc_meta.series,
                    release=doc_meta.release,
                    release_number=doc_meta.release_number,
                    version=doc_meta.version,
                    title=doc_meta.title,
                    clause_number=clause_number,
                    clause_title=block.elements[0].clause_title,
                    clause_path=block.elements[0].clause_path,
                    parent_clause=block.elements[0].parent_clause,
                    parent_title=block.elements[0].parent_title,
                    content_type=block.content_type,
                    text=text,
                    token_count=count_tokens(text),
                    chunk_index=index,
                    parent_chunk_id=parent_chunk_id,
                    is_annex=block.elements[0].is_annex,
                    is_table=block.content_type == ContentType.TABLE,
                    is_figure=any(e.is_figure for e in block.elements),
                    is_asn1=block.content_type == ContentType.ASN1,
                    table_data=block.elements[0].table_data if block.content_type == ContentType.TABLE else None,
                    source_file=doc_meta.source_file,
                    source_url=doc_meta.source_url,
                    source_locator=_source_locator(doc_meta, clause_number),
                    content_hash=_content_hash(text),
                )
            )

    logger.info("Generated {} chunks for {}", len(chunks), doc_meta.spec_number)
    return chunks


def _chunk_id(doc_meta: DocumentMetadata, clause_number: str, suffix: str) -> str:
    safe_clause = clause_number or "root"
    return f"{doc_meta.spec_number}_R{doc_meta.release_number}_{doc_meta.version}_{safe_clause}_{suffix}"


def _source_locator(doc_meta: DocumentMetadata, clause_number: str) -> str:
    return f"TS {doc_meta.spec_number} v{doc_meta.version}, Clause {clause_number or 'N/A'}"
