from __future__ import annotations

from pathlib import Path

from app.models.schema import ContentType
from ingestion.docx_parser import parse_docx
from ingestion.structure_parser import extract_structural_elements
from tests.docx_fixtures import build_sample_docx


def _elements(tmp_path: Path):
    docx_path = tmp_path / "sample.docx"
    build_sample_docx(docx_path)
    return extract_structural_elements(parse_docx(docx_path))


def test_builds_correct_clause_hierarchy(tmp_path: Path) -> None:
    elements = _elements(tmp_path)

    step_element = next(e for e in elements if "Step 1" in e.text)

    assert step_element.clause_number == "5.1.1"
    assert step_element.clause_title == "Registration procedure"
    assert step_element.clause_path == ["5", "5.1", "5.1.1"]
    assert step_element.parent_clause == "5.1"
    assert step_element.parent_title == "Overview"


def test_regex_alone_does_not_trigger_heading_detection(tmp_path: Path) -> None:
    # A body paragraph starting with something that *looks* clause-numbered
    # but isn't styled as a heading must not be treated as one.
    from ingestion.docx_parser import RawParagraph
    from ingestion.structure_parser import extract_structural_elements

    fake_elements = [
        RawParagraph(text="5.9.9 is referenced elsewhere in this document.", style_name="Normal", outline_level=None, numbering_id=None),
    ]
    result = extract_structural_elements(fake_elements)

    assert all(e.content_type != ContentType.HEADING for e in result)


def test_step_paragraphs_tagged_as_procedure(tmp_path: Path) -> None:
    elements = _elements(tmp_path)
    step = next(e for e in elements if "Step 1" in e.text)
    assert step.content_type == ContentType.PROCEDURE


def test_note_paragraph_tagged_as_note(tmp_path: Path) -> None:
    elements = _elements(tmp_path)
    note = next(e for e in elements if e.text.startswith("NOTE"))
    assert note.content_type == ContentType.NOTE


def test_table_element_carries_clause_context_and_table_data(tmp_path: Path) -> None:
    elements = _elements(tmp_path)
    table_element = next(e for e in elements if e.content_type == ContentType.TABLE)

    assert table_element.clause_number == "5.2"
    assert table_element.is_table is True
    assert table_element.table_data is not None
    assert table_element.table_data.headers == ["IE", "Presence", "Description"]


def test_asn1_block_isolated_as_atomic_element(tmp_path: Path) -> None:
    elements = _elements(tmp_path)
    asn1_elements = [e for e in elements if e.content_type == ContentType.ASN1]

    assert len(asn1_elements) == 1
    assert asn1_elements[0].is_asn1 is True
    assert "SEQUENCE" in asn1_elements[0].text
    assert asn1_elements[0].clause_number == "5.3"


def test_annex_flag_set_for_content_after_annex_heading(tmp_path: Path) -> None:
    elements = _elements(tmp_path)

    pre_annex = next(e for e in elements if "Step 1" in e.text)
    post_annex = next(e for e in elements if "change history" in e.text.lower())

    assert pre_annex.is_annex is False
    assert post_annex.is_annex is True
    assert post_annex.clause_number == "A"
