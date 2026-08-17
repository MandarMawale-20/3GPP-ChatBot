#!/usr/bin/env python3
"""CLI: download the latest Rel-18 archive for one or more specs.

Usage:
    python scripts/download.py 23.501
    python scripts/download.py 23.501 23.502 24.501
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


def _series_for_spec(spec_number: str) -> str:
    config = get_settings().corpus
    for group in config["documents"].values():
        for doc in group:
            if doc["spec_number"] == spec_number:
                return doc["series"]
    raise ValueError(f"{spec_number} is not in the corpus allowlist (configs/corpus.yaml)")


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="*", help="Spec numbers to download, e.g. 23.501")
    parser.add_argument("--all", action="store_true", help="Download every allowlisted document")
    args = parser.parse_args()

    if args.all:
        spec_numbers = get_settings().allowed_spec_numbers
    elif args.specs:
        spec_numbers = args.specs
    else:
        parser.print_help()
        return 1

    failures: list[str] = []
    for spec_number in spec_numbers:
        try:
            series = _series_for_spec(spec_number)
            logger.info("Starting download+ingest for TS {}", spec_number)
            chunks = download_and_process(spec_number, series)
            logger.info("{} -> {} chunks", spec_number, len(chunks))
        except (DownloadError, ValueError) as exc:
            logger.error("Failed to process {}: {}", spec_number, exc)
            failures.append(spec_number)

    if failures:
        logger.error("{} document(s) failed: {}", len(failures), ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
