"""Lightweight, regex-based query understanding for retrieval (no LLM).

The chatbot is release-controlled: a user query often names a specific 3GPP
specification ("TS 24.501", "section 23.502", "38.331"). Detecting that
spec number lets the retriever scope its metadata filter to the exact
document the user is asking about, which dramatically improves precision
over searching the whole corpus.

This is intentionally rule-based (regular expressions), not an LLM call:
it must be fast, deterministic, and free of any external dependency so it
can run on every `/chat` and `/search` request without latency or cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches 3GPP spec numbers in the forms:
#   - "24.501", "TS 24.501", "TS24.501", "3GPP TS 24.501"
#   - "23.502", "38.331"
# 3GPP spec numbers are a two-digit series, dot, two/three-digit number.
# The negative lookahead `(?!\.\d)` prevents matching version strings such
# as "18.9.0" (three dot-separated groups) or a spec followed by a version
# ("24.501 v18.9.0" yields "24.501" only). Clause numbers like "5.5.1" are
# safely ignored because they start with a single digit, failing the leading
# `\d{2}` requirement.
_SPEC_NUMBER_RE = re.compile(
    r"\b(?:TS\s*?)?(\d{2}\.\d{2,3})\b(?!\.\d)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractedQueryFilters:
    """Structured result of parse-free query analysis.

    `spec_number` is the detected 3GPP spec number (e.g. "24.501") and is
    None when no spec number could be confidently extracted from the query.
    """

    spec_number: str | None


def extract_query_filters(query: str) -> ExtractedQueryFilters:
    """Return any spec-number filter implied by the user's free-text query.

    The match is purely lexical. We return the first spec number that looks
    well-formed; 3GPP spec numbers are two-digit series, dot, two/three-digit
    number (e.g. 24.501, 23.502, 38.331). Version strings such as "18.9.0"
    are excluded because they have three dot-separated groups.

    Args:
        query: The raw user query string.

    Returns:
        ExtractedQueryFilters with `spec_number` set when a candidate is
        found, otherwise None.
    """
    if not query:
        return ExtractedQueryFilters(spec_number=None)

    match = _SPEC_NUMBER_RE.search(query)
    if not match:
        return ExtractedQueryFilters(spec_number=None)

    candidate = match.group(1)
    return ExtractedQueryFilters(spec_number=candidate)