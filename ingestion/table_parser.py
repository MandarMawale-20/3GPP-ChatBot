"""Table normalization and Markdown rendering.

1. Normalize OpenXML merge geometry (`RawTable` from docx_parser.py) into
   the canonical `TableData` model: vertically-merged continuation cells
   are filled with their inherited value, horizontal spans preserved via
   `colspan`.
2. Render the normalized table as GitHub-flavored Markdown for the
   retrieval-facing `Chunk.text` field.

Vertical merges are represented by filling forward the inherited text
into each continuation cell rather than using `rowspan > 1`: a filled
value is unambiguous for both Markdown (no native rowspan) and embedding
text.
"""

from __future__ import annotations

from app.models.schema import TableCell, TableData
from ingestion.docx_parser import RawTable


def normalize_table(raw: RawTable) -> TableData:
    """Convert OpenXML merge geometry into a flat, merge-resolved grid.

    First row is the header row (standard 3GPP convention). Column
    position is tracked explicitly because `gridSpan` means a row's cell
    count does not equal the table's column count.
    """
    if not raw.rows:
        return TableData(headers=[], rows=[])

    headers = _expand_header(raw.rows[0])

    # Per column index, the last non-continuation text seen; lets a
    # vertically-merged continuation cell inherit its value.
    column_last_text: dict[int, str] = {}

    data_rows: list[list[TableCell]] = []
    for raw_row in raw.rows[1:]:
        row_cells: list[TableCell] = []
        column_pointer = 0
        for raw_cell in raw_row:
            is_continuation = raw_cell.v_merge == "continue"
            if is_continuation:
                # Inherit rather than leave blank, so merged regions never
                # render as meaningless empty cells.
                text = column_last_text.get(column_pointer, "")
            else:
                text = raw_cell.text
                column_last_text[column_pointer] = text

            row_cells.append(
                TableCell(
                    text=text,
                    rowspan=1,  # merges are filled forward, not spanned
                    colspan=raw_cell.grid_span,
                    is_merged_continuation=is_continuation,
                )
            )
            column_pointer += raw_cell.grid_span
        data_rows.append(row_cells)

    return TableData(headers=headers, rows=data_rows)


def _expand_header(header_row: list) -> list[str]:
    """Flatten the header row into one string per physical column,
    duplicating text across horizontal merges so `len(headers)` equals the
    table's true column count (required by the `TableData` width check).
    """
    headers: list[str] = []
    for cell in header_row:
        headers.extend([cell.text] * cell.grid_span)
    return headers


def render_markdown_table(table: TableData) -> str:
    """Render a normalized table as GitHub-flavored Markdown (the
    retrieval-facing representation stored in `Chunk.text`). Colspan is
    rendered by repeating the cell's text across the spanned columns so
    the grid stays rectangular; Markdown has no merge syntax.
    """
    if not table.headers:
        return ""

    lines = [
        "| " + " | ".join(_escape_pipes(h) for h in table.headers) + " |",
        "| " + " | ".join("---" for _ in table.headers) + " |",
    ]
    for row in table.rows:
        cells: list[str] = []
        for cell in row:
            cells.extend([_escape_pipes(cell.text)] * cell.colspan)
        # Pad/truncate on edge-case width mismatch; rendering must never
        # crash the pipeline.
        cells = (cells + [""] * len(table.headers))[: len(table.headers)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _escape_pipes(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()
