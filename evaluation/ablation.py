#!/usr/bin/env python3
"""Retrieval ablation study (SRD §47): Experiments A-C.

    A. Dense only
    B. Dense + sparse (RRF fusion)
    C. Dense + sparse + reranker

(Experiments D and E — "+ evidence gate" and "+ citation validation" — are
answer-level, not retrieval-level, and are already covered by the metrics
`evaluate_generation.py` reports: unnecessary/false abstention rates
isolate the evidence gate's effect, and citation_accuracy isolates
citation validation's effect.)

Usage:
    python evaluation/ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.config import get_settings
from app.dependencies import get_embedding_provider, get_reranker
from app.logging_config import configure_logging
from app.retrieval.dense import dense_search
from app.retrieval.hybrid import build_release_spec_filter, reciprocal_rank_fusion
from app.retrieval.qdrant_store import get_client
from app.retrieval.sparse import sparse_search
from evaluation.evaluate_retrieval import DATASET_PATH, RESULTS_DIR, _is_relevant

TOP_K = 10


def _recall_at_k(chunks_per_question: list[list], questions: list[dict], k: int) -> float:
    hits = 0
    for chunks, question in zip(chunks_per_question, questions):
        top_k = chunks[:k]
        if any(_is_relevant(c, question["expected_spec"], question.get("expected_clause_prefix", "")) for c in top_k):
            hits += 1
    return hits / len(questions) if questions else 0.0


def run_ablation() -> dict:
    dataset = json.loads(DATASET_PATH.read_text())
    questions = [q for q in dataset["questions"] if q["answerable"]]
    release = dataset["release"]

    settings = get_settings().settings
    client = get_client(settings.qdrant_url, settings.qdrant_api_key)
    embedding_provider = get_embedding_provider()
    reranker = get_reranker()

    dense_only_results: list[list] = []
    hybrid_results: list[list] = []
    hybrid_reranked_results: list[list] = []

    for question in questions:
        query_filter = build_release_spec_filter(release)
        dense_vector = embedding_provider.embed_dense([question["query"]])[0]
        sparse_vector = embedding_provider.embed_sparse([question["query"]])[0]

        dense_hits = dense_search(client, settings.qdrant_collection, dense_vector, TOP_K, query_filter)
        sparse_hits = sparse_search(client, settings.qdrant_collection, sparse_vector, TOP_K, query_filter)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits])
        reranked = reranker.rerank(question["query"], fused, TOP_K)

        dense_only_results.append([r.chunk for r in dense_hits])
        hybrid_results.append([r.chunk for r in fused])
        hybrid_reranked_results.append([r.chunk for r in reranked])

    return {
        "A_dense_only": {"recall_at_10": _recall_at_k(dense_only_results, questions, 10)},
        "B_dense_plus_sparse": {"recall_at_10": _recall_at_k(hybrid_results, questions, 10)},
        "C_dense_plus_sparse_plus_reranker": {"recall_at_10": _recall_at_k(hybrid_reranked_results, questions, 10)},
        "num_questions": len(questions),
    }


def main() -> int:
    configure_logging()
    logger.info("Running retrieval ablation study...")
    results = run_ablation()

    for experiment, metrics in results.items():
        logger.info("   {}: {}", experiment, metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ablation_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results written to {}", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
