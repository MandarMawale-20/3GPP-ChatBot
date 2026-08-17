from __future__ import annotations

from pathlib import Path

from ingestion.docx_parser import RawParagraph, RawTable, parse_docx
from tests.docx_fixtures import build_sample_docx


def test_parse_docx_preserves_document_order(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    build_sample_docx(docx_path)

    elements = parse_docx(docx_path)
    types = [type(e).__name__ for e in elements]

    # The table must appear between the "5.2 Parameters" heading and the
    # "5.3 ASN.1 example" heading — i.e. document order (paragraph/table
    # interleaving) must be preserved, not paragraphs-then-tables.
    table_index = types.index("RawTable")
    heading_texts = [e.text for e in elements if isinstance(e, RawParagraph) and e.style_name == "Heading 2"]
    assert "5.2\tParameters" in heading_texts
    assert elements[table_index - 1].text == "5.2\tParameters"


def test_parse_docx_captures_heading_style(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    build_sample_docx(docx_path)

    elements = parse_docx(docx_path)
    heading = next(e for e in elements if isinstance(e, RawParagraph) and "General" in e.text)

    assert heading.style_name == "Heading 1"


def test_parse_docx_extracts_table_with_merge_metadata(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    build_sample_docx(docx_path)

    elements = parse_docx(docx_path)
    table = next(e for e in elements if isinstance(e, RawTable))

    assert table.rows[0][0].text == "IE"
    assert table.rows[1][0].v_merge == "restart"
    assert table.rows[2][0].v_merge == "continue"
    assert table.rows[2][0].text == ""  # continuation cell has no verbatim text


def test_parse_docx_captures_code_style_paragraphs(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    build_sample_docx(docx_path)

    elements = parse_docx(docx_path)
    tt_paragraphs = [e for e in elements if isinstance(e, RawParagraph) and e.style_name == "TT"]

    assert len(tt_paragraphs) == 5
    assert "SEQUENCE" in tt_paragraphs[0].text
