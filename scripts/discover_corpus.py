"""Automatic discovery of available 3GPP specifications.

Discovers specifications published in the official 3GPP Rel-18 directory and
compares them against the approved allowlist in configs/corpus.yaml. Discovery
is read-only: it never authorizes a document. corpus.yaml remains the sole
source of truth for what the downloader and ingestion pipeline may process;
a specification is only promoted to approved through an explicit `--add` after
a human reviews the candidate list.

The release-isolation guarantee (SRD §37, Decision 2) is preserved end to end:
the requested release is validated against corpus.yaml, and only archives whose
release-letter code matches that release are considered.

Reuses the directory-fetch, release-letter, and filename-decoding utilities
from ingestion.downloader rather than reimplementing them.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from loguru import logger

from app.config import get_settings
from ingestion.downloader import (
    RELEASE_LETTERS,
    base36_digit_to_int,
    fetch_directory_listing,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_YAML = PROJECT_ROOT / "configs" / "corpus.yaml"

# Series directories to scan under the Rel-18 repository root. These are
# series, not individual spec numbers — discovery derives spec numbers from
# the actual archive filenames it finds.
DEFAULT_SERIES = ["23", "24", "29", "33", "38"]

# Robust archive filename pattern. Handles:
#   - 5-digit spec codes (e.g. 24501) and an optional part suffix
#     (e.g. 38101-1 for TS 38.101-1), and
#   - release-letter + base-36 major/minor version digits (e.g. i90, ic0).
# This deliberately does not assume a fixed digit count before the version
# code: the part suffix means a filename can have a variable shape.
_ARCHIVE_RE = re.compile(
    r"^(?P<digits>\d{5,6})"
    r"(?:-(?P<part>\d+))?"
    r"-(?P<letter>[a-z])(?P<major>[0-9a-z])(?P<minor>[0-9a-z])\.(?P<ext>zip|docx|doc)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DiscoveredSpec:
    """A specification found in the official directory for the target release."""

    spec_number: str
    series: str
    version: str  # decoded, e.g. "18.12.0"
    filename: str
    _sort: tuple[int, int]  # (major, minor) for latest-version selection


def parse_archive_name(filename: str, release_number: int) -> DiscoveredSpec | None:
    """Parse a single archive filename into a DiscoveredSpec.

    Returns None if the filename is not a spec archive or belongs to a
    different release than the one requested (release isolation).
    """
    match = _ARCHIVE_RE.match(filename)
    if not match:
        return None

    expected_letter = RELEASE_LETTERS.get(release_number)
    letter = match.group("letter").lower()
    if expected_letter is not None and letter != expected_letter:
        # Belongs to a different release; never mix releases.
        return None

    digits = match.group("digits")
    series = digits[:2]
    body = digits[2:]
    spec_number = f"{series}.{body}"
    part = match.group("part")
    if part:
        spec_number = f"{spec_number}-{part}"

    major = base36_digit_to_int(match.group("major"))
    minor = base36_digit_to_int(match.group("minor"))
    version = f"{release_number}.{major}.{minor}"

    return DiscoveredSpec(
        spec_number=spec_number,
        series=series,
        version=version,
        filename=filename,
        _sort=(major, minor),
    )


def parse_listing(html: str, release_number: int) -> dict[str, DiscoveredSpec]:
    """Extract the latest-version DiscoveredSpec for each spec in a directory page."""
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, DiscoveredSpec] = {}

    for anchor in soup.find_all("a", href=True):
        filename = anchor["href"].rsplit("/", 1)[-1]
        spec = parse_archive_name(filename, release_number)
        if spec is None:
            continue
        existing = found.get(spec.spec_number)
        if existing is None or spec._sort > existing._sort:
            found[spec.spec_number] = spec

    return found


def discover_specs(release: str = "Rel-18", series: list[str] | None = None) -> dict[str, DiscoveredSpec]:
    """Fetch the official Rel-18 directories and return discovered specs.

    Reads the repository root and release/version mapping from corpus.yaml so
    discovery is consistent with the approved allowlist configuration.
    """
    config = get_settings()
    if release != config.corpus.get("release"):
        raise ValueError(
            f"Release mismatch: requested {release!r} but corpus.yaml allowlist "
            f"is for {config.corpus.get('release')!r}. Discovery will not mix releases."
        )

    release_number = config.corpus.get("release_number")
    if release_number is None:
        raise ValueError("corpus.yaml is missing 'release_number'.")

    repo_root = config.corpus.get("sources", {}).get("repository_root")
    if not repo_root:
        raise ValueError("corpus.yaml sources.repository_root is not configured.")

    series_list = series or DEFAULT_SERIES
    result: dict[str, DiscoveredSpec] = {}
    for s in series_list:
        url = f"{repo_root.rstrip('/')}/{s}_series/"
        logger.info("Discovering specifications from {}", url)
        html = fetch_directory_listing(url)
        result.update(parse_listing(html, release_number))

    return result


def load_approved_specs() -> list[str]:
    """Return the spec_numbers currently present in corpus.yaml (approved)."""
    if not CORPUS_YAML.exists():
        return []
    corpus = yaml.safe_load(CORPUS_YAML.read_text(encoding="utf-8")) or {}
    documents = corpus.get("documents", {}) or {}
    return [doc["spec_number"] for group in documents.values() for doc in group]


def parse_add_token(token: str) -> tuple[str, str | None]:
    """Split an --add token into (spec_number, title). Title may be absent."""
    if "=" in token:
        spec, title = token.split("=", 1)
        return spec.strip(), title.strip() or None
    return token.strip(), None


def build_additions(tokens: list[str], discovered: dict[str, DiscoveredSpec]) -> list[tuple[str, str, str]]:
    """Resolve --add tokens into (spec, title, series) tuples.

    Fails clearly if a spec is not verifiable in the official directory or if
    no title is available (titles are never invented).
    """
    additions: list[tuple[str, str, str]] = []
    for token in tokens:
        spec, title = parse_add_token(token)
        if spec not in discovered:
            raise ValueError(
                f"Specification {spec} was not found in the official Rel-18 "
                f"directory; an unverified document cannot be approved."
            )
        series = discovered[spec].series
        if not title:
            raise ValueError(
                f"No title available for {spec}. Pass it explicitly with "
                f"--add {spec}=<official title>. Titles are never invented."
            )
        additions.append((spec, title, series))
    return additions


def add_to_corpus_yaml(additions: list[tuple[str, str, str]], path: Path = CORPUS_YAML) -> None:
    """Append approved specifications to the extended group of corpus.yaml.

    Performs a targeted text insertion so existing structure, ordering, and
    comments are preserved as much as possible. Existing entries are never
    modified or removed.
    """
    approved = set(load_approved_specs_from(path))
    new_entries = [
        (spec, title, series) for spec, title, series in additions if spec not in approved
    ]
    if not new_entries:
        logger.info("No new specifications to add; corpus.yaml unchanged.")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    insert_idx = len(lines)
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and line.split(":", 1)[0] == "sources":
            insert_idx = i
            break

    block_lines = []
    for spec, title, series in new_entries:
        block_lines.append(f'    - spec_number: "{spec}"')
        block_lines.append(f'      title: "{title}"')
        block_lines.append(f'      series: "{series}"')
    block = "\n" + "\n".join(block_lines)

    lines.insert(insert_idx, block)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Added {} specification(s) to {}", len(new_entries), path)


def load_approved_specs_from(path: Path) -> list[str]:
    if not path.exists():
        return []
    corpus = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    documents = corpus.get("documents", {}) or {}
    return [doc["spec_number"] for group in documents.values() for doc in group]


def _print_discovery(discovered: dict[str, DiscoveredSpec], approved: list[str], missing_only: bool) -> None:
    approved_set = set(approved)
    by_series: dict[str, list[DiscoveredSpec]] = {}
    for spec in discovered.values():
        if missing_only and spec.spec_number in approved_set:
            continue
        by_series.setdefault(spec.series, []).append(spec)

    if not by_series:
        logger.info("No specifications to display.")
        return

    for series in sorted(by_series):
        logger.info("Series {}:", series)
        for spec in sorted(by_series[series], key=lambda s: s.spec_number):
            status = "approved" if spec.spec_number in approved_set else "new / not approved"
            logger.info("  {}  ({} {})  [{}]", spec.spec_number, spec.version, spec.filename, status)


def main() -> int:
    from app.logging_config import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="Rel-18", help="Target release (must match corpus.yaml)")
    parser.add_argument("--missing", action="store_true", help="Show only specs absent from corpus.yaml")
    parser.add_argument(
        "--add",
        nargs="+",
        metavar="SPEC[=TITLE]",
        help="Approve verified specs, e.g. --add 23.503='System architecture ...'",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help="Override the series directories to scan (default: 23 24 29 33 38)",
    )
    args = parser.parse_args()

    try:
        discovered = discover_specs(release=args.release, series=args.series)
    except ValueError as exc:
        logger.error("Discovery failed: {}", exc)
        return 1

    approved = load_approved_specs()

    if args.add:
        try:
            additions = build_additions(args.add, discovered)
        except ValueError as exc:
            logger.error("Cannot add specifications: {}", exc)
            return 1
        add_to_corpus_yaml(additions)
        logger.info("Approved specifications written to corpus.yaml.")
        return 0

    _print_discovery(discovered, approved, missing_only=args.missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())