#!/usr/bin/env python3
"""CLI: download the latest archive for one or more specs, per release.

The corpus allowlist (`configs/corpus.yaml`) is multi-release: a spec number
may appear in more than one release. `--all` processes every allowlisted
document across every enabled release; a named spec is resolved within a
single release (default: TARGET_RELEASE, or --release).

Usage:
    python scripts/download.py 23.501
    python scripts/download.py 23.501 --release Rel-17
    python scripts/download.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.config import get_settings
from app.logging_config import configure_logging
from ingestion.downloader import DownloadError
from ingestion.pipeline import download_and_process


def _doc_for_spec(spec_number: str, release: str) -> dict:
    """Resolve a spec number within one release's allowlist."""
    for d in get_settings().allowed_documents(release=release):
        if d["spec_number"] == spec_number:
            return d
    raise ValueError(
        f"{spec_number} is not in the {release} allowlist (configs/corpus.yaml). "
        "Approve it first with: scripts/discover_corpus.py --add ..."
    )


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="*", help="Spec numbers to download, e.g. 23.501")
    parser.add_argument(
        "--release",
        default=None,
        help="Release scope for named specs (default: TARGET_RELEASE). "
        "Ignored with --all (which spans every enabled release).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download every allowlisted document across all enabled releases",
    )
    args = parser.parse_args()

    if args.all:
        docs = get_settings().allowed_documents()  # all enabled releases
    elif args.specs:
        release = args.release or get_settings().default_release
        docs = [_doc_for_spec(spec, release) for spec in args.specs]
    else:
        parser.print_help()
        return 1

    failures: list[str] = []
    for d in docs:
        spec_number = d["spec_number"]
        series = d["series"]
        release = d["release"]
        try:
            logger.info("Starting download+ingest for TS {} ({})", spec_number, release)
            chunks = download_and_process(spec_number, series, release=release)
            logger.info("{} [{}] -> {} chunks", spec_number, release, len(chunks))
        except (DownloadError, ValueError) as exc:
            logger.error("Failed to process {} ({}): {}", spec_number, release, exc)
            failures.append(f"{spec_number}@{release}")

    if failures:
        logger.error("{} document(s) failed: {}", len(failures), ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
