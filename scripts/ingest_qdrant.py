#!/usr/bin/env python3
"""CLI: embed every chunk in data/processed/*.jsonl and upsert into Qdrant.

Usage:
    python scripts/ingest_qdrant.py
    python scripts/ingest_qdrant.py --spec 24.501
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.config import get_settings
from app.logging_config import configure_logging
from app.retrieval.embeddings import BGEM3EmbeddingProvider
from app.retrieval.qdrant_store import ensure_collection, get_client, upsert_chunks
from ingestion.pipeline import PROCESSED_DIR, read_jsonl


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", help="Only ingest one spec's JSONL file, e.g. 24.501")
    args = parser.parse_args()

    config = get_settings().settings

    jsonl_files = (
        [PROCESSED_DIR / f"{args.spec}.jsonl"] if args.spec else sorted(PROCESSED_DIR.glob("*.jsonl"))
    )
    jsonl_files = [p for p in jsonl_files if p.exists()]
    if not jsonl_files:
        logger.error("No JSONL files found in {}", PROCESSED_DIR)
        return 1

    logger.info("Loading embedding model...")
    embedding_provider = BGEM3EmbeddingProvider(model_name=config.embedding_model)

    client = get_client(config.qdrant_url, config.qdrant_api_key)
    ensure_collection(client, config.qdrant_collection)

    total = 0
    for path in jsonl_files:
        chunks = read_jsonl(path)
        logger.info("Ingesting {} ({} chunks)", path.name, len(chunks))
        total += upsert_chunks(
            client,
            config.qdrant_collection,
            chunks,
            embedding_provider,
            batch_size=config.embedding_batch_size,
        )

    logger.info("Ingested {} chunks total into '{}'", total, config.qdrant_collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
