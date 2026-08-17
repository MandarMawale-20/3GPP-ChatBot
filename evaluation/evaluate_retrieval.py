#!/usr/bin/env python3
"""Retrieval evaluation: Recall@5/10/20, Context Precision (SRD §45).

Runs each answerable benchmark question through the retriever directly
(bypassing the API/LLM) and checks whether at least one retrieved chunk
comes from the expected spec (and, if given, the expected clause prefix).

Usage:
    python evaluation/evaluate_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.dependencies import get_retriever
from app.logging_config import configure_logging

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _is_relevant(chunk, expected_spec: str | None, expected_clause_prefix: str) -> bool:
    if expected_spec is None or chunk.spec_number != expected_spec:
        return False
    if expected_clause_prefix and not chunk.clause_number.startswith(expected_clause_prefix):
        return False
    return True


def evaluate_retrieval() -> dict:
    dataset = json.loads(DATASET_PATH.read_text())
    questions = [q for q in dataset["questions"] if q["answerable"]]
    retriever = get_retriever()

    k_values = [5, 10, 20]
    hits_at_k = {k: 0 for k in k_values}
    precision_at_5_sum = 0.0

    for question in questions:
        retrieved = retriever.retrieve(question["query"], dataset["release"], rerank_top_k=max(k_values))
        chunks = [r.chunk for r in retrieved]

        for k in k_values:
            top_k_chunks = chunks[:k]
            hit = any(
                _is_relevant(c, question["expected_spec"], question.get("expected_clause_prefix", ""))
                for c in top_k_chunks
            )
            if hit:
                hits_at_k[k] += 1

        top_5 = chunks[:5]
        relevant_in_top_5 = sum(
            1 for c in top_5 if _is_relevant(c, question["expected_spec"], question.get("expected_clause_prefix", ""))
        )
        precision_at_5_sum += relevant_in_top_5 / max(len(top_5), 1)

    n = len(questions)
    results = {
        f"recall_at_{k}": hits_at_k[k] / n if n else 0.0 for k in k_values
    }
    results["context_precision_at_5"] = precision_at_5_sum / n if n else 0.0
    results["num_questions"] = n
    return results


def main() -> int:
    configure_logging()
    logger.info("Running retrieval evaluation...")
    results = evaluate_retrieval()

    for metric, value in results.items():
        logger.info("   {}: {}", metric, value)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "retrieval_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to {}", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
