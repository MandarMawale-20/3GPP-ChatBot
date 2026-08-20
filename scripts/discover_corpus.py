"""Automatic discovery of available 3GPP specifications.

Discovers specifications published in the official 3GPP directory for a release
and compares them against the approved allowlist in `configs/corpus.yaml`.
Discovery is read-only: it never authorizes a document. corpus.yaml remains
the sole source of truth for what the downloader and ingestion pipeline may
process; a specification is only promoted to approved through an explicit
`--add` after a human reviews the candidate list.

The release-isolation guarantee (SRD §37, Decision 2) is preserved end to end:
the requested release is validated against corpus.yaml (must be an *enabled*
release), and only archives whose release-letter code matches that release are
considered — Rel-17 ('h') and Rel-18 ('i') archives never mix within a single
release's pipeline.

Multi-release: `configs/corpus.yaml` carries an independent allowlist per
release (`releases.<release>`), so more than one release can be indexed at
once. `discover_specs(release)` scopes to one release; `discover_all()` browses
every enabled release (the latter is what a browse-all UI calls).

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from ingestion.downloader import (
    RELEASE_LETTERS,
    base36_digit_to_int,
    fetch_directory_listing,
)

CORPUS_YAML = PROJECT_ROOT / "configs" / "corpus.yaml"

# Series directories to scan under a release repository root. These are
# series, not individual spec numbers — discovery derives spec numbers from
# the actual archive filenames it finds. These series are shared across
# recent 3GPP releases; use --series to scan additional ones.
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


def discover_specs(
    release: str = "Rel-18", series: list[str] | None = None
) -> dict[str, DiscoveredSpec]:
    """Fetch the official <release> directories and return discovered specs.

    Reads the repository root and release/version mapping from the per-release
    config in corpus.yaml, so discovery is consistent with the approved
    allowlist for that release.
    """
    config = get_settings()
    cfg = config.release_config(release)  # raises if release is absent/disabled
    release_number = cfg["release_number"]
    repo_root = cfg["sources"]["repository_root"]

    series_list = series or DEFAULT_SERIES
    result: dict[str, DiscoveredSpec] = {}
    for s in series_list:
        url = f"{repo_root.rstrip('/')}/{s}_series/"
        logger.info("Discovering specifications for {} from {}", release, url)
        html = fetch_directory_listing(url)
        result.update(parse_listing(html, release_number))

    return result


def discover_all(series: list[str] | None = None) -> dict[str, dict[str, DiscoveredSpec]]:
    """Browse the official directories for every enabled release.

    Returns `{release: {spec_number: DiscoveredSpec}}`. Used by a browse-all UI
    to present all releases at once; each per-release result is independently
    release-isolated (never mixed at the archive level).
    """
    config = get_settings()
    all_specs: dict[str, dict[str, DiscoveredSpec]] = {}
    for release in config.enabled_releases:
        all_specs[release] = discover_specs(release=release, series=series)
    return all_specs


def fetch_available_releases() -> list[str]:
    """Return every release directory published on the official 3GPP FTP,
    ascending by release number, e.g. ``['Rel-8', ..., 'Rel-20']``.

    Used by the Corpus Manager to offer a *live* dropdown of all releases
    instead of the static enabled set in corpus.yaml. Parses the ``latest/``
    index page for ``Rel-<n>`` folder links.
    """
    ROOT = "https://www.3gpp.org/ftp/specs/latest/"
    html = fetch_directory_listing(ROOT)
    soup = BeautifulSoup(html, "html.parser")
    found: list[int] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].rstrip("/")
        name = href.rsplit("/", 1)[-1]
        m = re.match(r"^Rel-(\d+)$", name)
        if m:
            found.append(int(m.group(1)))
    # Ascending, stable. The UI may reverse for display.
    return [f"Rel-{n}" for n in sorted(found)]


def discover_release(
    release: str, series: list[str] | None = None
) -> dict[str, DiscoveredSpec]:
    """Fetch the official directory for ANY release, even one not yet present
    or enabled in corpus.yaml.

    Differs from :func:`discover_specs` (which enforces that the release be
    enabled/configured): this backs the Corpus Manager's live dropdown, where a
    human may browse and approve a release that has no config block yet.
    Resolution order for the release number / repository root:

      1. the per-release block in corpus.yaml (if present & parseable), else
      2. derived from the 3GPP live FTP naming (``Rel-<n>`` -> ``n``, repo
         under ``latest/Rel-<n>/``).

    Returns ``{}`` if the release cannot be resolved or its directory is
    unreachable. Release isolation is enforced exactly as in ``discover_specs``
    (the archive filename-letter code still must match the release).
    """
    # 1. Prefer the configured block (release_number + repository_root).
    try:
        cfg = get_settings().release_config(release)
        release_number = cfg["release_number"]
        repo_root = cfg["sources"]["repository_root"]
    except Exception:
        # 2. Fallback derivation from the live FTP convention.
        m = re.match(r"^Rel-(\d+)$", release)
        if not m:
            logger.error("Cannot resolve release {!r} without a corpus.yaml block.", release)
            return {}
        release_number = int(m.group(1))
        repo_root = f"https://www.3gpp.org/ftp/specs/latest/{release}/"

    series_list = series or DEFAULT_SERIES
    result: dict[str, DiscoveredSpec] = {}
    for s in series_list:
        url = f"{repo_root.rstrip('/')}/{s}_series/"
        try:
            html = fetch_directory_listing(url)
        except Exception as exc:
            logger.warning("Could not fetch {}: {}", url, exc)
            continue
        result.update(parse_listing(html, release_number))
    return result


def ensure_release_present(release: str, path: Path = CORPUS_YAML) -> None:
    """Create an enabled release block in corpus.yaml if it doesn't exist yet.

    Lets the Corpus Manager approve specs for a live-discovered release that
    has no config block (e.g. an unconfigured Rel-19). If the block exists but
    is disabled, it is flipped to ``enabled: true`` so the new documents can be
    indexed. Existing entries are never modified or removed.
    """
    m = re.match(r"^Rel-(\d+)$", release)
    if not m:
        raise ValueError(f"Malformed release identifier: {release!r}")
    release_number = int(m.group(1))

    corpus = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    releases = corpus.get("releases", {}) or {}
    if release in releases:
        if not releases[release].get("enabled", True):
            _set_enabled(path, release, True)
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    rel_idx = next((i for i, ln in enumerate(lines) if ln.strip() == "releases:"), None)
    if rel_idx is None:
        raise ValueError("No 'releases:' key found in corpus.yaml.")
    # Insert the new block (with a leading blank line) right after `releases:`.
    lines[rel_idx + 1 : rel_idx + 1] = _format_release_block(release, release_number)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Created release block {} in {}", release, path)


def _set_enabled(path: Path, release: str, enabled: bool) -> None:
    """Flip the `enabled:` flag for one release block (text edit, no reformat)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = _release_block_indices(lines, release)
    for i in range(start, end):
        if lines[i].strip().startswith("enabled:"):
            indent = len(lines[i]) - len(lines[i].lstrip(" "))
            lines[i] = f"{' ' * indent}enabled: {str(enabled).lower()}"
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_release_block(release: str, release_number: int) -> list[str]:
    """YAML lines for a freshly-created enabled release block (2-space indent)."""
    repo_root = f"https://www.3gpp.org/ftp/specs/latest/{release}/"
    return [
        "",
        f"  {release}:",
        f"    release_number: {release_number}",
        "    enabled: true",
        "    sources:",
        '      portal: "https://portal.3gpp.org/"',
        f'      repository_root: "{repo_root}"',
        '      forge: "https://forge.3gpp.org/swagger/tools/parser.html"',
        "    documents:",
        "      core: []",
        "      extended: []",
    ]


def load_approved_specs(release: str) -> list[str]:
    """Return the spec_numbers currently present in corpus.yaml for one release."""
    return load_approved_specs_from(CORPUS_YAML, release)


def load_approved_specs_from(path: Path, release: str) -> list[str]:
    """spec_numbers approved under `releases.<release>` in a corpus file."""
    if not path.exists():
        return []
    corpus = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    releases = corpus.get("releases", {}) or {}
    if release not in releases:
        return []
    documents = releases[release].get("documents", {}) or {}
    return [doc["spec_number"] for group in documents.values() for doc in group]


def parse_add_token(token: str) -> tuple[str, str | None]:
    """Split an --add token into (spec_number, title). Title may be absent."""
    if "=" in token:
        spec, title = token.split("=", 1)
        return spec.strip(), title.strip() or None
    return token.strip(), None


def build_additions(
    tokens: list[str], discovered: dict[str, DiscoveredSpec], release: str
) -> list[tuple[str, str, str]]:
    """Resolve --add tokens into (spec, title, series) tuples.

    Fails clearly if a spec is not verifiable in the official directory or if
    no title is available (titles are never invented).
    """
    additions: list[tuple[str, str, str]] = []
    for token in tokens:
        spec, title = parse_add_token(token)
        if spec not in discovered:
            raise ValueError(
                f"Specification {spec} was not found in the official {release} "
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


def add_to_corpus_yaml(
    additions: list[tuple[str, str, str]], release: str, path: Path = CORPUS_YAML
) -> None:
    """Append approved specifications to the `extended` group of one release.

    Performs a targeted, comment-preserving text insertion into
    `releases.<release>.documents.extended` so existing structure, ordering,
    and comments are preserved. Existing entries are never modified or removed.
    """
    # Only enabled releases may be edited. A disabled release (e.g. the
    # Rel-16/19/20 scaffolding) has no live pipeline, so approving documents
    # into it is incoherent — discovery can't even run for a disabled release.
    corpus = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    releases = corpus.get("releases", {}) or {}
    if release not in releases:
        raise ValueError(f"Release {release!r} is not present in {path}.")
    if not releases[release].get("enabled", False):
        raise ValueError(
            f"Release {release!r} is disabled in {path}; set `enabled: true` "
            "before adding documents."
        )
    approved = set(load_approved_specs_from(path, release))
    new_entries = [
        (spec, title, series) for spec, title, series in additions if spec not in approved
    ]
    if not new_entries:
        logger.info("No new specifications to add to {}; corpus.yaml unchanged.", release)
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    insert_idx, indent = _extended_insertion_point(lines, release)

    if indent == "[]":
        # The list was `      extended: []`; replace the inline `[]` with a
        # real block list carrying the new entries. We build the full block
        # first and splice it once: inserting each entry at a fixed index in a
        # loop would *prepend* them and reverse their order.
        lines[insert_idx] = f"{' ' * 6}extended:"
        new_block: list[str] = []
        for spec, title, series in new_entries:
            new_block.extend(_format_extended_entry(spec, title, series))
        lines[insert_idx + 1 : insert_idx + 1] = new_block
    else:
        # Real block list: splice the new entries in at the end of the list.
        for spec, title, series in reversed(new_entries):
            for line in reversed(_format_extended_entry(spec, title, series)):
                lines[insert_idx:insert_idx] = [line]

    # Persist the in-memory mutation. This is the single write-back point for an
    # allowlist write, so every code path (inline-empty expansion, block-append)
    # lands on disk atomically.
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Added {} specification(s) to {} [{}]", len(new_entries), path, release)


def _format_extended_entry(spec: str, title: str, series: str) -> list[str]:
    """YAML list-item lines for one allowlisted document (8/10-space indent)."""
    return [
        f'        - spec_number: "{spec}"',
        f'          title: "{title}"',
        f'          series: "{series}"',
    ]


def _release_block_indices(lines: list[str], release: str) -> tuple[int, int]:
    """Return (start, end) line range for the `  <release>:` block under releases:."""
    key = f"  {release}:"
    start = next((i for i, ln in enumerate(lines) if ln == key), None)
    if start is None:
        raise ValueError(f"Release {release!r} is not present in corpus.yaml 'releases'.")
    end = len(lines)
    for j in range(start + 1, end):
        raw = lines[j]
        if raw.strip():  # non-blank
            indent = len(raw) - len(raw.lstrip(" "))
            if indent <= 2:  # sibling release key or top-level key
                end = j
                break
    return start, end


def _extended_insertion_point(lines: list[str], release: str) -> tuple[int, str]:
    """Locate where to append to `releases.<release>.documents.extended`.

    Returns `(index, sentinel)` where:
      - sentinel == "[]" means the list is the inline `      extended: []` form
        (line at `index` is that line) and the caller should expand it;
      - otherwise `index` is the line position immediately *before which* new
        list items should be inserted (the end of the existing list, or right
        after the `extended:` header for an empty block list).
    """
    start, end = _release_block_indices(lines, release)
    # Match both `extended:` and `extended: []` (the inline-empty form that
    # this corpus uses for not-yet-populated releases).
    ext_idx = next(
        (i for i in range(start, end) if lines[i].strip().startswith("extended:")),
        None,
    )
    if ext_idx is None:
        raise ValueError(f"No 'extended' allowlist under release {release!r} in corpus.yaml.")

    # Inline empty form, e.g. `      extended: []`. The whole line is the
    # sentinel; the caller expands it into a real block list.
    if lines[ext_idx].strip() == "extended: []":
        return ext_idx, "[]"

    # Block-list form (`extended:` on its own line, or already populated).
    # Walk forward past existing list items (indent >= 8) to find the end.
    ins = ext_idx + 1
    for i in range(ext_idx + 1, end):
        raw = lines[i]
        if not raw.strip():
            continue  # preserve blank lines but don't stop counting the list
        indent = len(raw) - len(raw.lstrip(" "))
        if indent < 8:
            break  # left the list (e.g. next sibling / next release)
        ins = i + 1
    return ins, ""


def _print_discovery(
    discovered: dict[str, DiscoveredSpec], approved: list[str], missing_only: bool, release: str
) -> None:
    approved_set = set(approved)
    by_series: dict[str, list[DiscoveredSpec]] = {}
    for spec in discovered.values():
        if missing_only and spec.spec_number in approved_set:
            continue
        by_series.setdefault(spec.series, []).append(spec)

    if not by_series:
        logger.info("No specifications to display for {}.", release)
        return

    logger.info("Release {} — discovered/allowlist status:", release)
    for series in sorted(by_series):
        logger.info("  Series {}:", series)
        for spec in sorted(by_series[series], key=lambda s: s.spec_number):
            status = "approved" if spec.spec_number in approved_set else "new / not approved"
            logger.info("  {}  ({} {})  [{}]", spec.spec_number, spec.version, spec.filename, status)


def main() -> int:
    from app.logging_config import configure_logging

    configure_logging()
    config = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release",
        default=None,
        help="Target release (must be an enabled release in corpus.yaml). "
        f"Default: {config.default_release}.",
    )
    parser.add_argument("--missing", action="store_true", help="Show only specs absent from corpus.yaml for that release")
    parser.add_argument(
        "--add",
        nargs="+",
        metavar="SPEC[=TITLE]",
        help="Approve verified specs into the release's `extended` allowlist, "
        "e.g. --add 23.503='System architecture ...'.",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help="Override the series directories to scan (default: 23 24 29 33 38)",
    )
    args = parser.parse_args()

    release = args.release or config.default_release
    if release not in config.enabled_releases:
        logger.error(
            "Release {} is not enabled in corpus.yaml. Enabled: {}",
            release,
            config.enabled_releases,
        )
        return 1

    try:
        discovered = discover_specs(release=release, series=args.series)
    except ValueError as exc:
        logger.error("Discovery failed: {}", exc)
        return 1

    approved = load_approved_specs(release)

    if args.add:
        try:
            additions = build_additions(args.add, discovered, release=release)
        except ValueError as exc:
            logger.error("Cannot add specifications: {}", exc)
            return 1
        add_to_corpus_yaml(additions, release=release)
        logger.info("Approved specifications written to corpus.yaml [{}].", release)
        return 0

    _print_discovery(discovered, approved, missing_only=args.missing, release=release)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
