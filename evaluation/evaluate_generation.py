#!/usr/bin/env python3
"""Generation/safety evaluation (SRD §45): citation accuracy, abstention
accuracy, and a hallucination-rate proxy based on claim verification.

Runs the full grounded-generation pipeline (retrieval -> evidence gate ->
LLM -> verification -> citation validation) for every benchmark question
and scores the outcomes against the dataset's `answerable` labels.

Usage:
    python evaluation/evaluate_generation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.citations.validator import validate_citations
from app.dependencies import get_generator, get_retriever
from app.generation.verifier import verify_answer
from app.logging_config import configure_logging

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def evaluate_generation() -> dict:
    dataset = json.loads(DATASET_PATH.read_text())
    questions = dataset["questions"]

    retriever = get_retriever()
    generator = get_generator()

    correct_abstentions = 0
    false_answers = 0  # answered when it should have abstained
    unnecessary_abstentions = 0  # abstained when a real answer was expected
    citation_valid_count = 0
    citation_total_count = 0
    hallucination_flags = 0
    answered_count = 0

    per_question_results = []

    for question in questions:
        retrieved = retriever.retrieve(question["query"], dataset["release"])
        result = generator.answer(question["query"], retrieved)

        expected_answerable = question["answerable"]

        if result.abstained and not expected_answerable:
            correct_abstentions += 1
        elif result.abstained and expected_answerable:
            unnecessary_abstentions += 1
        elif not result.abstained and not expected_answerable:
            false_answers += 1
        elif not result.abstained and expected_answerable:
            answered_count += 1
            chunks = [r.chunk for r in retrieved]
            citation_check = validate_citations(result.answer, result.citations)
            citation_total_count += 1
            if citation_check.valid:
                citation_valid_count += 1

            verification = verify_answer(result.answer, chunks)
            if not verification.passed:
                hallucination_flags += 1

        per_question_results.append(
            {
                "id": question["id"],
                "category": question["category"],
                "expected_answerable": expected_answerable,
                "abstained": result.abstained,
            }
        )

    n = len(questions)
    results = {
        "num_questions": n,
        "correct_abstentions": correct_abstentions,
        "false_answer_rate": false_answers / n if n else 0.0,
        "unnecessary_abstention_rate": unnecessary_abstentions / n if n else 0.0,
        "citation_accuracy": citation_valid_count / citation_total_count if citation_total_count else None,
        "hallucination_rate": hallucination_flags / answered_count if answered_count else None,
        "per_question": per_question_results,
    }
    return results


def main() -> int:
    configure_logging()
    logger.info("Running generation evaluation (requires a populated Qdrant + LLM API key)...")
    results = evaluate_generation()

    for metric, value in results.items():
        if metric != "per_question":
            logger.info("   {}: {}", metric, value)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "generation_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to {}", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
