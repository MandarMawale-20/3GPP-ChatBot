#!/usr/bin/env python3
"""CLI: parse and chunk a local DOCX file that's already on disk.

Use this when you have a 3GPP DOCX file already downloaded (manually, or
from a prior `download.py` run) and just want to (re-)run parsing and
chunking on it — e.g. for testing the parser against a real sample file
without touching the network.

Usage:
    python scripts/preprocess.py path/to/24501-i90.docx \\
        --spec 24.501 --series 24 --version 18.9.0 \\
        --title "Non-Access-Stratum (NAS) protocol for 5G System (5GS)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.config import get_settings
from app.logging_config import configure_logging
from ingestion.pipeline import PROCESSED_DIR, process_local_document, write_jsonl


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx_path", type=Path, help="Path to a local DOCX file")
    parser.add_argument("--spec", required=True, help="Spec number, e.g. 24.501")
    parser.add_argument("--series", required=True, help="Series, e.g. 24")
    parser.add_argument("--version", required=True, help="Version, e.g. 18.9.0")
    parser.add_argument("--title", required=True, help="Specification title")
    parser.add_argument("--source-url", default="", help="Original URL, if known")
    args = parser.parse_args()

    if not args.docx_path.exists():
        logger.error("File not found: {}", args.docx_path)
        return 1

    config = get_settings()
    release = config.settings.target_release
    release_number = int(release.split("-")[1])

    chunks = process_local_document(
        docx_path=args.docx_path,
        spec_number=args.spec,
        series=args.series,
        release=release,
        release_number=release_number,
        version=args.version,
        title=args.title,
        source_file=args.docx_path.name,
        source_url=args.source_url,
    )

    write_jsonl(chunks, PROCESSED_DIR / f"{args.spec}.jsonl")
    logger.info("Done: {} chunks", len(chunks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
