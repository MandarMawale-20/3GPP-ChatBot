from __future__ import annotations

from ingestion.docx_parser import RawCell, RawTable
from ingestion.table_parser import normalize_table, render_markdown_table


def test_normalize_table_expands_header_colspan() -> None:
    raw = RawTable(
        rows=[
            [RawCell(text="Merged Header", grid_span=2, v_merge=None), RawCell(text="C", grid_span=1, v_merge=None)],
            [
                RawCell(text="a1", grid_span=1, v_merge=None),
                RawCell(text="a2", grid_span=1, v_merge=None),
                RawCell(text="a3", grid_span=1, v_merge=None),
            ],
        ]
    )

    table = normalize_table(raw)

    assert table.headers == ["Merged Header", "Merged Header", "C"]


def test_normalize_table_fills_vertical_merge_continuation() -> None:
    raw = RawTable(
        rows=[
            [RawCell(text="H1", grid_span=1, v_merge=None), RawCell(text="H2", grid_span=1, v_merge=None)],
            [RawCell(text="shared", grid_span=1, v_merge="restart"), RawCell(text="x", grid_span=1, v_merge=None)],
            [RawCell(text="", grid_span=1, v_merge="continue"), RawCell(text="y", grid_span=1, v_merge=None)],
        ]
    )

    table = normalize_table(raw)

    assert table.rows[0][0].text == "shared"
    assert table.rows[1][0].text == "shared"  # inherited, not blank
    assert table.rows[1][0].is_merged_continuation is True
    assert table.rows[0][0].is_merged_continuation is False


def test_normalize_table_empty_table_returns_empty() -> None:
    table = normalize_table(RawTable(rows=[]))
    assert table.headers == []
    assert table.rows == []


def test_render_markdown_table_produces_valid_grid() -> None:
    raw = RawTable(
        rows=[
            [RawCell(text="IE", grid_span=1, v_merge=None), RawCell(text="Presence", grid_span=1, v_merge=None)],
            [RawCell(text="5GMM cause", grid_span=1, v_merge=None), RawCell(text="M", grid_span=1, v_merge=None)],
        ]
    )
    table = normalize_table(raw)

    markdown = render_markdown_table(table)
    lines = markdown.splitlines()

    assert lines[0] == "| IE | Presence |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 5GMM cause | M |"


def test_render_markdown_table_escapes_pipe_characters() -> None:
    raw = RawTable(
        rows=[
            [RawCell(text="A", grid_span=1, v_merge=None)],
            [RawCell(text="value | with pipe", grid_span=1, v_merge=None)],
        ]
    )
    table = normalize_table(raw)
    markdown = render_markdown_table(table)

    assert "\\|" in markdown
    assert "||" not in markdown.replace("| ---", "")  # no unescaped malformed pipes


def test_render_markdown_table_never_produces_meaningless_empty_row() -> None:
    # Regression guard for SRD §13's explicit "shall not result in
    # meaningless |||" requirement.
    raw = RawTable(
        rows=[
            [RawCell(text="H1", grid_span=1, v_merge=None), RawCell(text="H2", grid_span=1, v_merge=None)],
            [RawCell(text="shared", grid_span=1, v_merge="restart"), RawCell(text="x", grid_span=1, v_merge=None)],
            [RawCell(text="", grid_span=1, v_merge="continue"), RawCell(text="y", grid_span=1, v_merge=None)],
        ]
    )
    table = normalize_table(raw)
    markdown = render_markdown_table(table)

    assert "| |" not in markdown
    assert "shared" in markdown.splitlines()[-1]
