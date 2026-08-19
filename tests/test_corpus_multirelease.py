"""Multi-release correctness for the corpus/config/discovery layer.

These tests are network-free: they exercise the in-process, release-aware
config API and the comment-preserving corpus writer against a temp copy of
configs/corpus.yaml, plus the filename release-isolation guard in the parser.
They pin the behavior that lets Rel-17 and Rel-18 coexist in one corpus
without cross-contamination.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from scripts.discover_corpus import (  # noqa: E402
    _extended_insertion_point,
    _release_block_indices,
    add_to_corpus_yaml,
    load_approved_specs_from,
    parse_archive_name,
)

CORPUS_YAML = PROJECT_ROOT / "configs" / "corpus.yaml"


# ---------------------------------------------------------------------------
# AppConfig multi-release API
# ---------------------------------------------------------------------------

def test_enabled_releases_includes_both() -> None:
    config = get_settings()
    enabled = config.enabled_releases
    assert "Rel-17" in enabled
    assert "Rel-18" in enabled
    # Disabled scaffolding releases are excluded.
    assert "Rel-16" not in enabled
    assert "Rel-19" not in enabled


def test_default_release_is_enabled() -> None:
    config = get_settings()
    assert config.default_release in config.enabled_releases


def test_release_config_exposes_release_number_and_sources() -> None:
    config = get_settings()
    r17 = config.release_config("Rel-17")
    assert r17["release_number"] == 17
    assert r17["sources"]["repository_root"].endswith("/Rel-17/")
    r18 = config.release_config("Rel-18")
    assert r18["release_number"] == 18


def test_release_config_rejects_disabled_and_unconfigured() -> None:
    config = get_settings()
    with pytest.raises(ValueError):
        config.release_config("Rel-16")  # disabled scaffolding
    with pytest.raises(ValueError):
        config.release_config("Rel-99")  # not present at all


def test_allowed_documents_scoped_to_release() -> None:
    config = get_settings()
    # Rel-17 is enabled but ships an empty allowlist (scaffolding, awaiting
    # `--add`); Rel-18 carries documents.
    assert config.allowed_documents("Rel-17") == []
    r18_docs = config.allowed_documents("Rel-18")
    assert all(d["release"] == "Rel-18" for d in r18_docs)
    assert all(d["release_number"] == 18 for d in r18_docs)
    assert {d["spec_number"] for d in r18_docs} >= {"23.501", "24.501"}

    # All-releases mode returns docs across every enabled release. Rel-17 is
    # enabled but empty, so it contributes no documents — yet both releases
    # are addressable and isolated.
    all_docs = config.allowed_documents()
    assert {d["release"] for d in all_docs} == {"Rel-18"}
    assert "Rel-17" in config.enabled_releases


# ---------------------------------------------------------------------------
# Archive filename release-isolation guard
# ---------------------------------------------------------------------------

def test_parse_archive_name_accepts_matching_release_letter() -> None:
    # 'i' is the Rel-18 letter; 9/0 are base-36 minor/major digits.
    spec = parse_archive_name("24501-i90.docx", release_number=18)
    assert spec is not None
    assert spec.spec_number == "24.501"
    assert spec.series == "24"
    assert spec.version == "18.9.0"


def test_parse_archive_name_rejects_wrong_release_letter() -> None:
    # 'h' is Rel-17's letter; an '-hd0' archive must NOT parse for Rel-18.
    assert parse_archive_name("24501-hd0.docx", release_number=18) is None
    # And the converse: a 'i90' archive must not parse for Rel-17.
    assert parse_archive_name("24501-i90.docx", release_number=17) is None


def test_parse_archive_name_part_suffix() -> None:
    spec = parse_archive_name("38101-1-i90.docx", release_number=18)
    assert spec is not None
    assert spec.spec_number == "38.101-1"


# ---------------------------------------------------------------------------
# Comment-preserving, comment-writing corpus editor (multi-release writes)
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "corpus.yaml"
    shutil.copy(CORPUS_YAML, dst)
    return dst


def test_add_to_corpus_appends_to_block_list_release(corpus_copy: Path) -> None:
    """Rel-18 has a populated `extended:` block; appends into it."""
    before = load_approved_specs_from(corpus_copy, "Rel-18")
    assert "23.501" in before  # core
    assert "38.300" in before  # existing extended

    add_to_corpus_yaml(
        [("99.981", "A Totally Fictional Spec", "99")], "Rel-18", path=corpus_copy
    )

    after = load_approved_specs_from(corpus_copy, "Rel-18")
    assert "99.981" in after
    assert "23.501" in after  # untouched
    assert "38.300" in after  # untouched
    # Whole file must still be valid YAML after the text splice.
    yaml.safe_load(corpus_copy.read_text(encoding="utf-8"))
    # Header comments preserved (the write is a targeted text edit, not a round-trip).
    assert "single source of truth" in corpus_copy.read_text(encoding="utf-8")


def test_add_to_corpus_expands_inline_empty_release(corpus_copy: Path) -> None:
    """Rel-17 ships `extended: []`; the writer must expand it to a block list."""
    assert load_approved_specs_from(corpus_copy, "Rel-17") == []
    add_to_corpus_yaml(
        [("23.503", "System architecture for the 5G System (5GS)", "23")],
        "Rel-17",
        path=corpus_copy,
    )
    after = load_approved_specs_from(corpus_copy, "Rel-17")
    assert after == ["23.503"]


def test_add_to_corpus_preserves_order_on_inline_empty(corpus_copy: Path) -> None:
    """Expanding `extended: []` must keep the supplied order (not reverse it).

    Regression: inserting multiple entries one at a time at a fixed splice
    index prepends each, which reversed the order on `extended: []` expansion.
    """
    assert load_approved_specs_from(corpus_copy, "Rel-17") == []
    add_to_corpus_yaml(
        [
            ("23.001", "AAA Title", "23"),
            ("23.002", "BBB Title", "23"),
            ("23.003", "CCC Title", "23"),
        ],
        "Rel-17",
        path=corpus_copy,
    )
    after = load_approved_specs_from(corpus_copy, "Rel-17")
    assert after == ["23.001", "23.002", "23.003"]


def test_add_to_corpus_is_idempotent(corpus_copy: Path) -> None:
    lines_before = corpus_copy.read_text(encoding="utf-8").splitlines()
    add_to_corpus_yaml(
        [("38.300", "NR and NG-RAN Overall Description", "38")], "Rel-18", path=corpus_copy
    )
    lines_after = corpus_copy.read_text(encoding="utf-8").splitlines()
    assert lines_after == lines_before  # no-op: spec already approved


def test_add_to_corpus_rejects_release_not_in_file(corpus_copy: Path) -> None:
    with pytest.raises(ValueError):
        add_to_corpus_yaml(
            [("1.1", "x", "01")], "Rel-19", path=corpus_copy  # disabled -> no extended block accessible
        )


def test_release_block_indices_and_insertion_point() -> None:
    lines = CORPUS_YAML.read_text(encoding="utf-8").splitlines()
    start, end = _release_block_indices(lines, "Rel-18")
    assert lines[start] == "  Rel-18:"
    # The block ends at the next indent<=2 line — the comment that precedes
    # the disabled Rel-16 scaffolding — rather than running to EOF.
    assert end < len(lines)
    assert lines[end].lstrip().startswith("#")
    # Rel-17 (the first release block) ends at its sibling release, Rel-18.
    start17, end17 = _release_block_indices(lines, "Rel-17")
    assert lines[start17] == "  Rel-17:"
    assert lines[end17] == "  Rel-18:"
    # Rel-18 extended is a populated block list (no inline `[]` sentinel).
    idx, sentinel = _extended_insertion_point(lines, "Rel-18")
    assert sentinel == ""  # block-append form
    # Rel-17 extended is the inline-empty form.
    _, sentinel17 = _extended_insertion_point(lines, "Rel-17")
    assert sentinel17 == "[]"
