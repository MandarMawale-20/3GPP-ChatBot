#!/usr/bin/env python3
"""CLI: validate every data/processed/*.jsonl file and report corpus stats.

Every line is re-parsed through the `Chunk` pydantic model, so a malformed
or schema-drifted record is caught here rather than silently reaching
Qdrant ingestion.

Usage:
    python scripts/build_jsonl.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.logging_config import configure_logging
from ingestion.pipeline import PROCESSED_DIR, read_jsonl


def main() -> int:
    configure_logging()

    jsonl_files = sorted(PROCESSED_DIR.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL files found in {}", PROCESSED_DIR)
        return 1

    total_chunks = 0
    content_type_counts: Counter[str] = Counter()
    invalid_files: list[str] = []

    for path in jsonl_files:
        try:
            chunks = read_jsonl(path)
        except Exception as exc:  # noqa: BLE001 — report and continue, don't crash the report
            logger.error("Invalid JSONL in {}: {}", path.name, exc)
            invalid_files.append(path.name)
            continue

        total_chunks += len(chunks)
        for chunk in chunks:
            content_type_counts[chunk.content_type.value] += 1

        tables = sum(1 for c in chunks if c.is_table)
        asn1 = sum(1 for c in chunks if c.is_asn1)
        logger.info("{}: {} chunks ({} tables, {} ASN.1)", path.name, len(chunks), tables, asn1)

    logger.info("Total chunks across corpus: {}", total_chunks)
    for content_type, count in content_type_counts.most_common():
        logger.info("   {}: {}", content_type, count)

    if invalid_files:
        logger.error("{} file(s) failed validation: {}", len(invalid_files), ", ".join(invalid_files))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
