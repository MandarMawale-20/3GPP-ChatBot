"""ASN.1 block detection.

3GPP specifications (notably TS 38.331 and other RAN protocol specs) embed
ASN.1 module definitions as runs of consecutive paragraphs. Detection
combines two signals:

1. Paragraph style — 3GPP's Word template uses a fixed-width style
   (commonly "TT", typewriter text) for code-like content.
2. Lexical signal — ASN.1 syntax markers (`::=`, `SEQUENCE`, `CHOICE`,
   `OPTIONAL`, `ENUMERATED`) unlikely to co-occur in ordinary prose.

Consecutive matching paragraphs are grouped into a single atomic block so
ASN.1 definitions are not arbitrarily split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ingestion.docx_parser import RawParagraph

# Paragraph styles 3GPP documents commonly use for monospaced/code content.
_CODE_STYLE_NAMES = {"tt", "code", "asn1", "programcode", "programlisting"}

# Lexical tokens that strongly indicate ASN.1 syntax rather than prose.
_ASN1_TOKEN_RE = re.compile(
    r"(::=|\bSEQUENCE\b|\bCHOICE\b|\bENUMERATED\b|\bOPTIONAL\b|\bOCTET STRING\b"
    r"|\bBIT STRING\b|\bBOOLEAN\b|\bINTEGER\s*\(|\bDEFAULT\b\s+\w|BEGIN\s*$|^END\s*$)"
)


@dataclass(frozen=True)
class Asn1Block:
    text: str
    start_index: int  # index into the original paragraph list
    end_index: int  # inclusive


def _looks_like_asn1(paragraph: RawParagraph) -> bool:
    style = paragraph.style_name.strip().lower().replace(" ", "")
    if style in _CODE_STYLE_NAMES:
        # Style is the stronger signal; any non-trivial code-styled
        # paragraph is a candidate.
        return bool(paragraph.text.strip())
    return bool(_ASN1_TOKEN_RE.search(paragraph.text))


def detect_asn1_blocks(paragraphs: list[RawParagraph]) -> list[Asn1Block]:
    """Scan a paragraph sequence and group consecutive ASN.1-like
    paragraphs into atomic blocks.

    Short runs of blank lines inside an active block are tolerated, since
    ASN.1 definitions frequently contain blank lines for readability.
    """
    blocks: list[Asn1Block] = []
    block_start: int | None = None
    block_lines: list[str] = []
    blank_run = 0

    def _flush(end_index: int) -> None:
        nonlocal block_start, block_lines
        if block_start is not None and any(line.strip() for line in block_lines):
            blocks.append(
                Asn1Block(text="\n".join(block_lines).rstrip(), start_index=block_start, end_index=end_index)
            )
        block_start = None
        block_lines = []

    for index, paragraph in enumerate(paragraphs):
        is_match = _looks_like_asn1(paragraph)
        is_blank = not paragraph.text.strip()

        if is_match:
            if block_start is None:
                block_start = index
            block_lines.append(paragraph.text)
            blank_run = 0
        elif is_blank and block_start is not None:
            # Tolerate a short run of blank lines inside an active block.
            block_lines.append("")
            blank_run += 1
            if blank_run > 2:
                _flush(index - blank_run)
                blank_run = 0
        else:
            _flush(index - 1)
            blank_run = 0

    if block_start is not None:
        _flush(len(paragraphs) - 1)

    return blocks
