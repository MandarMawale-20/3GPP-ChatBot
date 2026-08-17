from __future__ import annotations

import pytest

from app.retrieval.query_preprocessor import extract_query_filters


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What is timer T3510 in TS 24.501?", "24.501"),
        ("How is a PDU session established in 23.502", "23.502"),
        ("Explain clause 5.5.1 of 38.331", "38.331"),
        ("Compare 33.501 and 33.501 security", "33.501"),
        ("Tell me about 24.501 v18.9.0 registration", "24.501"),
        ("What does TS24.501 say about NAS?", "24.501"),
        ("3GPP TS 23.502 registration procedure", "23.502"),
        ("What is 5QI in 23.501?", "23.501"),
    ],
)
def test_extracts_well_formed_spec_number(query: str, expected: str) -> None:
    result = extract_query_filters(query)
    assert result.spec_number == expected


@pytest.mark.parametrize(
    "query",
    [
        "What is the role of the AMF in the 5G System architecture?",
        "How is a PDU session established?",
        "What is timer T3510 and when is it started?",
        "",  # empty
        "What is the default value of timer T3502?",  # no spec number mentioned
        "In Release 18, what changes were introduced to NR-U?",  # release, not spec
    ],
)
def test_no_spec_number_extracted(query: str) -> None:
    result = extract_query_filters(query)
    assert result.spec_number is None


def test_version_strings_do_not_match() -> None:
    # "18.9.0" is a three-group version string, not a spec number.
    result = extract_query_filters("What changed in version 18.9.0 of the spec?")
    assert result.spec_number is None


def test_clause_numbers_do_not_match() -> None:
    # "5.5.1" and "9.2" are single-digit-prefixed clause numbers.
    result = extract_query_filters("What is in clause 5.5.1?")
    assert result.spec_number is None

    result2 = extract_query_filters("Refer to section 9.2 for details")
    assert result2.spec_number is None