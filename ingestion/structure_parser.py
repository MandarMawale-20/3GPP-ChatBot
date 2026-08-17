"""Clause hierarchy + structural element extraction.

Combines the raw paragraph/table stream from `docx_parser` with heading/
clause detection (style + numbering + regex; regex alone is not trusted),
ASN.1 block grouping, table normalization, and procedure/note/figure
caption tagging, into a flat ordered list of `StructuralElement`s tagged
with their full clause path. Direct input to the chunker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from app.models.schema import ContentType, TableData
from ingestion.asn1_parser import detect_asn1_blocks
from ingestion.docx_parser import RawElement, RawParagraph, RawTable
from ingestion.table_parser import normalize_table, render_markdown_table

# Numeric clause heading, e.g. "5.3.5.1 Registration procedure" or "5 General".
_NUMERIC_HEADING_RE = re.compile(r"^(?P<number>\d+(?:\.\d+){0,6})\s+(?P<title>.+)$")

# Annex heading, e.g. "Annex A (normative): Change history" or "Annex A.1 ...".
_ANNEX_HEADING_RE = re.compile(
    r"^Annex\s+(?P<number>[A-Z](?:\.\d+){0,4})\s*(?:\([^)]*\))?\s*:?\s*(?P<title>.*)$",
    re.IGNORECASE,
)

_STEP_RE = re.compile(r"^\s*Step\s*\d+\s*[:.]", re.IGNORECASE)
_NOTE_RE = re.compile(r"^\s*NOTE\s*\d*\s*[:.]", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*Figure\s+[A-Za-z0-9.]+[-:]?\s*.+$", re.IGNORECASE)

_HEADING_STYLE_RE = re.compile(r"heading|title", re.IGNORECASE)


@dataclass
class StructuralElement:
    content_type: ContentType
    text: str
    clause_number: str
    clause_title: str
    clause_path: list[str] = field(default_factory=list)
    parent_clause: str = ""
    parent_title: str = ""
    is_annex: bool = False
    is_asn1: bool = False
    is_figure: bool = False
    is_table: bool = False
    is_normative: bool = True
    table_data: TableData | None = None


def _is_heading(paragraph: RawParagraph) -> tuple[str, str] | None:
    """Return (clause_number, title) if this paragraph is a genuine clause
    or annex heading, else None.

    Requires BOTH a structural signal (Word heading style / outline level)
    AND a lexical match against the clause-numbering pattern; regex alone
    is not sufficient since body text can start with "5.3.5.1" in a cross
    reference.
    """
    text = paragraph.text.strip()
    if not text:
        return None

    structurally_heading = (
        bool(_HEADING_STYLE_RE.search(paragraph.style_name))
        or paragraph.outline_level is not None
    )

    annex_match = _ANNEX_HEADING_RE.match(text)
    if annex_match:
        # Annex titles are often not styled as "Heading N" in 3GPP
        # templates, so the "Annex " prefix alone is accepted as the
        # structural signal.
        return annex_match.group("number").upper(), annex_match.group("title").strip()

    numeric_match = _NUMERIC_HEADING_RE.match(text)
    if numeric_match and structurally_heading:
        return numeric_match.group("number"), numeric_match.group("title").strip()

    return None


def _content_type_for_paragraph(text: str) -> ContentType:
    if _STEP_RE.match(text):
        return ContentType.PROCEDURE
    if _NOTE_RE.match(text):
        return ContentType.NOTE
    if _FIGURE_CAPTION_RE.match(text):
        return ContentType.FIGURE
    return ContentType.PARAGRAPH


class _ClauseTracker:
    """Maintains the current clause path/title stack as headings are
    encountered, and whether we've entered the annex section (once an
    annex heading is seen, every subsequent clause is an annex clause).
    """

    def __init__(self) -> None:
        self.path: list[str] = []
        self.titles: dict[str, str] = {}
        self.in_annex = False

    def enter_heading(self, number: str, title: str, is_annex_heading: bool) -> None:
        if is_annex_heading:
            self.in_annex = True

        depth = number.count(".") + 1
        self.path = self.path[: depth - 1] + [number]
        self.titles[number] = title

    @property
    def clause_number(self) -> str:
        return self.path[-1] if self.path else ""

    @property
    def clause_title(self) -> str:
        return self.titles.get(self.clause_number, "")

    @property
    def parent_clause(self) -> str:
        return self.path[-2] if len(self.path) >= 2 else ""

    @property
    def parent_title(self) -> str:
        return self.titles.get(self.parent_clause, "")


def extract_structural_elements(elements: list[RawElement]) -> list[StructuralElement]:
    """Walk the raw paragraph/table stream and produce clause-tagged
    structural elements, ready for chunking.
    """
    tracker = _ClauseTracker()
    output: list[StructuralElement] = []

    # Buffer consecutive RawParagraphs so ASN.1 detection can examine a
    # contiguous run at a time; a table interruption ends the run.
    paragraph_buffer: list[RawParagraph] = []

    def _flush_paragraph_buffer() -> None:
        if not paragraph_buffer:
            return
        _process_paragraph_run(paragraph_buffer, tracker, output)
        paragraph_buffer.clear()

    for element in elements:
        if isinstance(element, RawParagraph):
            paragraph_buffer.append(element)
        elif isinstance(element, RawTable):
            _flush_paragraph_buffer()
            _process_table(element, tracker, output)

    _flush_paragraph_buffer()
    return output


def _process_paragraph_run(
    paragraphs: list[RawParagraph], tracker: _ClauseTracker, output: list[StructuralElement]
) -> None:
    asn1_blocks = detect_asn1_blocks(paragraphs)
    asn1_covered_indices: set[int] = set()
    for block in asn1_blocks:
        asn1_covered_indices.update(range(block.start_index, block.end_index + 1))

    asn1_by_start = {block.start_index: block for block in asn1_blocks}

    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]

        if index in asn1_by_start:
            block = asn1_by_start[index]
            output.append(
                StructuralElement(
                    content_type=ContentType.ASN1,
                    text=block.text,
                    clause_number=tracker.clause_number,
                    clause_title=tracker.clause_title,
                    clause_path=list(tracker.path),
                    parent_clause=tracker.parent_clause,
                    parent_title=tracker.parent_title,
                    is_annex=tracker.in_annex,
                    is_asn1=True,
                )
            )
            index = block.end_index + 1
            continue

        if index in asn1_covered_indices:
            # Already emitted as part of an ASN.1 block starting earlier
            # in this run; guards against overlap bugs.
            index += 1
            continue

        heading = _is_heading(paragraph)
        if heading:
            number, title = heading
            is_annex_heading = bool(_ANNEX_HEADING_RE.match(paragraph.text.strip()))
            tracker.enter_heading(number, title, is_annex_heading)
            output.append(
                StructuralElement(
                    content_type=ContentType.HEADING,
                    text=paragraph.text.strip(),
                    clause_number=tracker.clause_number,
                    clause_title=tracker.clause_title,
                    clause_path=list(tracker.path),
                    parent_clause=tracker.parent_clause,
                    parent_title=tracker.parent_title,
                    is_annex=tracker.in_annex,
                )
            )
            index += 1
            continue

        text = paragraph.text.strip()
        if text:
            content_type = _content_type_for_paragraph(text)
            is_figure = content_type == ContentType.FIGURE
            output.append(
                StructuralElement(
                    content_type=content_type,
                    # Figures are never interpreted, only flagged.
                    text=f"[FIGURE_PRESENT] {text}" if is_figure else text,
                    clause_number=tracker.clause_number,
                    clause_title=tracker.clause_title,
                    clause_path=list(tracker.path),
                    parent_clause=tracker.parent_clause,
                    parent_title=tracker.parent_title,
                    is_annex=tracker.in_annex,
                    is_figure=is_figure,
                )
            )
        index += 1


def _process_table(raw_table: RawTable, tracker: _ClauseTracker, output: list[StructuralElement]) -> None:
    if not raw_table.rows:
        return

    try:
        table_data = normalize_table(raw_table)
        markdown = render_markdown_table(table_data)
    except Exception as exc:  # noqa: BLE001 — a malformed table must degrade, never abort the document
        # Preserve raw text rather than crash ingestion. TABLE chunks must
        # carry valid table_data, so this becomes a plain-text paragraph
        # instead: still retrievable, without structured table metadata.
        logger.warning(
            "Malformed table at clause {} — falling back to raw text ({})",
            tracker.clause_number or "root",
            exc,
        )
        output.append(
            StructuralElement(
                content_type=ContentType.PARAGRAPH,
                text=_raw_table_fallback_text(raw_table),
                clause_number=tracker.clause_number,
                clause_title=tracker.clause_title,
                clause_path=list(tracker.path),
                parent_clause=tracker.parent_clause,
                parent_title=tracker.parent_title,
                is_annex=tracker.in_annex,
                is_table=False,
            )
        )
        return

    output.append(
        StructuralElement(
            content_type=ContentType.TABLE,
            text=markdown,
            clause_number=tracker.clause_number,
            clause_title=tracker.clause_title,
            clause_path=list(tracker.path),
            parent_clause=tracker.parent_clause,
            parent_title=tracker.parent_title,
            is_annex=tracker.in_annex,
            is_table=True,
            table_data=table_data,
        )
    )


def _raw_table_fallback_text(raw_table: RawTable) -> str:
    """Best-effort text preservation for a table that couldn't be
    normalized: join each row's cell text so content remains searchable.
    """
    lines = []
    for row in raw_table.rows:
        cell_texts = [cell.text for cell in row if cell.text]
        if cell_texts:
            lines.append(" | ".join(cell_texts))
    return "\n".join(lines)
