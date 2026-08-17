from __future__ import annotations

from ingestion.asn1_parser import detect_asn1_blocks
from ingestion.docx_parser import RawParagraph


def _para(text: str, style: str = "Normal") -> RawParagraph:
    return RawParagraph(text=text, style_name=style, outline_level=None, numbering_id=None)


def test_detects_single_asn1_block_by_style() -> None:
    paragraphs = [
        _para("Ordinary prose before the block."),
        _para("RRCSetupRequest ::= SEQUENCE {", style="TT"),
        _para("    ue-Identity InitialUE-Identity,", style="TT"),
        _para("}", style="TT"),
        _para("Ordinary prose after the block."),
    ]

    blocks = detect_asn1_blocks(paragraphs)

    assert len(blocks) == 1
    assert blocks[0].start_index == 1
    assert blocks[0].end_index == 3
    assert "SEQUENCE" in blocks[0].text


def test_detects_asn1_block_by_lexical_signal_without_code_style() -> None:
    paragraphs = [
        _para("Normal prose."),
        _para("Foo ::= SEQUENCE { bar INTEGER }"),
        _para("More normal prose."),
    ]

    blocks = detect_asn1_blocks(paragraphs)

    assert len(blocks) == 1
    assert blocks[0].start_index == 1


def test_tolerates_short_blank_gap_inside_block() -> None:
    paragraphs = [
        _para("Foo ::= SEQUENCE {", style="TT"),
        _para("", style="TT"),
        _para("bar INTEGER", style="TT"),
        _para("}", style="TT"),
    ]

    blocks = detect_asn1_blocks(paragraphs)

    assert len(blocks) == 1
    assert blocks[0].start_index == 0
    assert blocks[0].end_index == 3


def test_does_not_flag_ordinary_prose() -> None:
    paragraphs = [
        _para("The UE shall send a REGISTRATION REQUEST message to the AMF."),
        _para("The AMF shall then initiate the authentication procedure."),
    ]

    blocks = detect_asn1_blocks(paragraphs)

    assert blocks == []


def test_empty_paragraph_list_returns_no_blocks() -> None:
    assert detect_asn1_blocks([]) == []
