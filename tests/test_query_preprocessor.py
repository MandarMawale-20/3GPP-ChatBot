from __future__ import annotations

from app.retrieval.query_preprocessor import extract_query_filters


def test_extracts_spec_number_from_query() -> None:
    """A user query naming a 3GPP spec number is parsed into spec_number."""
    result = extract_query_filters("What is timer T3510 in TS 24.501?")
    assert result.spec_number == "24.501"
