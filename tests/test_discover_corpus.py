"""Tests for the automatic corpus discovery utility.

Network access is mocked throughout — these tests never touch the live 3GPP
directory. They verify filename parsing, release isolation, the
approved/missing comparison, refusal to add unverifiable specs, and that
corpus.yaml's allowlist is preserved.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts import discover_corpus as dc


def _listing_html(filenames: list[str]) -> str:
    links = "".join(f'<a href="{f}">{f}</a>' for f in filenames)
    return f"<html><body>{links}</body></html>"


# --- filename parsing -----------------------------------------------------

def test_parse_23501_ic0() -> None:
    spec = dc.parse_archive_name("23501-ic0.zip", release_number=18)
    assert spec is not None
    assert spec.spec_number == "23.501"
    assert spec.series == "23"
    assert spec.version == "18.12.0"  # c -> 12


def test_parse_handles_base36_versions() -> None:
    for name, major, minor in [
        ("23501-i90.zip", 9, 0),
        ("23501-ia0.zip", 10, 0),
        ("23501-ib0.zip", 11, 0),
        ("23501-ic0.zip", 12, 0),
    ]:
        spec = dc.parse_archive_name(name, release_number=18)
        assert spec is not None
        assert spec.version == f"18.{major}.{minor}"


def test_parse_38101_part_suffix() -> None:
    spec = dc.parse_archive_name("38101-1-ic0.zip", release_number=18)
    assert spec is not None
    assert spec.spec_number == "38.101-1"
    assert spec.series == "38"


def test_parse_rejects_other_release_letter() -> None:
    # Rel-18 letter is 'i'; 'h' belongs to Rel-17.
    spec = dc.parse_archive_name("23501-h90.zip", release_number=18)
    assert spec is None


def test_parse_ignores_non_archive_links() -> None:
    assert dc.parse_archive_name("../", release_number=18) is None
    assert dc.parse_archive_name("index.html", release_number=18) is None


def test_parse_listing_keeps_latest_version() -> None:
    html = _listing_html(
        [
            "23501-i90.zip",
            "23501-ia0.zip",  # newer (10 > 9)
            "23501-ic0.zip",  # newest (12)
            "23502-ie0.zip",
            "24501-i90.zip",
        ]
    )
    found = dc.parse_listing(html, release_number=18)
    assert found["23.501"].version == "18.12.0"
    assert set(found) == {"23.501", "23.502", "24.501"}


# --- discover_specs (mocked network) --------------------------------------

@pytest.fixture
def fake_config(monkeypatch) -> None:
    class _Corpus:
        corpus = {
            "release": "Rel-18",
            "release_number": 18,
            "sources": {"repository_root": "https://example.org/Rel-18/"},
        }

    monkeypatch.setattr(dc, "get_settings", lambda: _Corpus())


def test_discover_specs_aggregates_series(monkeypatch, fake_config) -> None:
    series_pages = {
        "23": _listing_html(["23501-ic0.zip", "23502-ie0.zip", "23503-ic0.zip"]),
        "24": _listing_html(["24501-i90.zip"]),
        "29": _listing_html(["29244-ic0.zip"]),
        "33": _listing_html([]),
        "38": _listing_html(["38300-ic0.zip", "38331-ic0.zip", "38101-1-ic0.zip"]),
    }

    def fake_fetch(url: str) -> str:
        for s, html in series_pages.items():
            if f"/{s}_series/" in url:
                return html
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(dc, "fetch_directory_listing", fake_fetch)

    discovered = dc.discover_specs(release="Rel-18")
    specs = set(discovered)
    assert "23.501" in specs
    assert "23.503" in specs
    assert "24.501" in specs
    assert "29.244" in specs
    assert "38.300" in specs
    assert "38.331" in specs
    assert "38.101-1" in specs


def test_discover_rejects_release_mismatch(monkeypatch, fake_config) -> None:
    def fake_fetch(url: str) -> str:
        return _listing_html(["23501-ic0.zip"])

    monkeypatch.setattr(dc, "fetch_directory_listing", fake_fetch)

    with pytest.raises(ValueError):
        dc.discover_specs(release="Rel-19")


# --- approved / missing comparison ----------------------------------------

def test_load_approved_specs(tmp_path, monkeypatch) -> None:
    yaml_text = textwrap.dedent(
        """
        documents:
          core:
            - spec_number: "23.501"
              title: t
              series: "23"
            - spec_number: "24.501"
              title: t
              series: "24"
          extended:
            - spec_number: "38.300"
              title: t
              series: "38"
        """
    )
    p = tmp_path / "corpus.yaml"
    p.write_text(yaml_text)

    approved = dc.load_approved_specs_from(p)
    assert approved == ["23.501", "24.501", "38.300"]


def test_missing_detection(fake_config) -> None:
    discovered = {
        "23.501": dc.DiscoveredSpec("23.501", "23", "18.12.0", "23501-ic0.zip", (12, 0)),
        "23.503": dc.DiscoveredSpec("23.503", "23", "18.12.0", "23503-ic0.zip", (12, 0)),
        "24.501": dc.DiscoveredSpec("24.501", "24", "18.9.0", "24501-i90.zip", (9, 0)),
    }
    approved = ["23.501", "24.501"]

    missing = {s for s in discovered if s not in set(approved)}
    assert missing == {"23.503"}


# --- refusing to add unverifiable specs -----------------------------------

def test_build_additions_rejects_unverified(fake_config) -> None:
    discovered = {
        "23.501": dc.DiscoveredSpec("23.501", "23", "18.12.0", "23501-ic0.zip", (12, 0)),
    }
    with pytest.raises(ValueError, match="not found"):
        dc.build_additions(["23.999=Title"], discovered)


def test_build_additions_requires_title(fake_config) -> None:
    discovered = {
        "23.503": dc.DiscoveredSpec("23.503", "23", "18.12.0", "23503-ic0.zip", (12, 0)),
    }
    with pytest.raises(ValueError, match="No title"):
        dc.build_additions(["23.503"], discovered)


def test_build_additions_ok(fake_config) -> None:
    discovered = {
        "23.503": dc.DiscoveredSpec("23.503", "23", "18.12.0", "23503-ic0.zip", (12, 0)),
    }
    additions = dc.build_additions(["23.503=Some Title"], discovered)
    assert additions == [("23.503", "Some Title", "23")]


# --- add_to_corpus_yaml preserves allowlist -------------------------------

BASE_YAML = textwrap.dedent(
    """
    release: Rel-18

    documents:
      core:
        - spec_number: "23.501"
          title: "Core 501"
          series: "23"
      extended:
        - spec_number: "38.300"
          title: "RRC overview"
          series: "38"

    sources:
      portal: "https://portal.3gpp.org/"
    """
).lstrip()


def test_add_to_corpus_yaml_appends_and_preserves(tmp_path) -> None:
    p = tmp_path / "corpus.yaml"
    p.write_text(BASE_YAML)

    additions = [("23.503", "Procedures B", "23"), ("29.500", "SBA", "29")]
    dc.add_to_corpus_yaml(additions, path=p)

    import yaml

    corpus = yaml.safe_load(p.read_text(encoding="utf-8"))
    all_specs = [d["spec_number"] for group in corpus["documents"].values() for d in group]
    assert "23.501" in all_specs  # original preserved
    assert "38.300" in all_specs  # original preserved
    assert "23.503" in all_specs  # added
    assert "29.500" in all_specs  # added

    # Original titles untouched.
    titles = {d["spec_number"]: d["title"] for group in corpus["documents"].values() for d in group}
    assert titles["23.501"] == "Core 501"


def test_add_to_corpus_yaml_skips_existing(tmp_path) -> None:
    p = tmp_path / "corpus.yaml"
    p.write_text(BASE_YAML)
    before = p.read_text(encoding="utf-8")

    # 23.501 already exists -> should be a no-op (idempotent).
    dc.add_to_corpus_yaml([("23.501", "Core 501", "23")], path=p)
    assert p.read_text(encoding="utf-8") == before


def test_allowlist_remains_authoritative(tmp_path, monkeypatch) -> None:
    """Discovery output never auto-approves; download pipeline still keys off
    corpus.yaml's allowed_spec_numbers (simulated here by load_approved_specs)."""
    p = tmp_path / "corpus.yaml"
    p.write_text(BASE_YAML)
    monkeypatch.setattr(dc, "CORPUS_YAML", p)

    approved = dc.load_approved_specs()
    assert approved == ["23.501", "38.300"]
    # A discovered-but-unadded spec is NOT in the approved list;
    # approval requires an explicit --add step.
    assert "23.503" not in approved
