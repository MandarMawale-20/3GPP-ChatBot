#!/usr/bin/env python3
"""Ragas-based LLM-judge evaluation: faithfulness + context precision.

Unlike `evaluation/evaluate_generation.py` (which uses deterministic rules to
score citations / abstention behaviour), this harness measures *answer
quality* with LLM-as-judge metrics from ragas:

  - faithfulness:      is the generated answer actually supported by the
                       retrieved contexts (no hallucination)?
  - context_precision: how relevant are the retrieved contexts to the query?
  - context_recall:    do the retrieved contexts cover a reference (gold)
                       answer?

The judge LLM is **Gemini** (`google-genai`) — never OpenAI. A reference
(gold) answer is generated for each benchmark question the first time it is
needed and cached to `evaluation/results/ragas_references.json`, so the
`context_recall` metric has a ground truth to compare against.

This intentionally uses NO OpenAI dependency. Extra packages required to run
the real (non-fake) path:

    pip install ragas langchain-google-genai

(ragas drives the metric computation; langchain-google-genai provides the
Gemini chat model that ragas uses as its judge.)

Usage:
    python evaluation/ragas_eval.py                 # real Gemini judge + RAG
    python evaluation/ragas_eval.py --fake          # no network; dumps dataset
    python evaluation/ragas_eval.py --limit 5       # only first 5 questions
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
REFERENCES_PATH = RESULTS_DIR / "ragas_references.json"


@dataclass
class QAPair:
    """One question with its system-produced answer and retrieved contexts."""

    id: str
    query: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    abstained: bool = False


class ReferenceJudge(Protocol):
    """Produces a gold/reference answer for a benchmark question."""

    def generate(self, question: str, expected_spec: str | None) -> str: ...


class GeminiReferenceJudge:
    """Reference-answer writer backed by Gemini (`google-genai`).

    Used to build the ground truth that the `context_recall` metric compares
    retrieved contexts against. It is separate from the chatbot's own LLM
    call so the judge cannot simply echo the chatbot's answer.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash") -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the ragas reference judge")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "google-genai is required for GeminiReferenceJudge. "
                "Install it with `pip install google-genai`."
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model_name = model_name

    def generate(self, question: str, expected_spec: str | None) -> str:
        spec_hint = f" (primarily TS {expected_spec})" if expected_spec else ""
        prompt = (
            "You are a 3GPP standards expert. Write a concise, factually precise "
            "reference answer to the following question about 3GPP Release-18"
            f"{spec_hint}. Answer strictly from the standard's technical content; "
            "include the relevant spec number and clause where useful. Do not add "
            "information beyond the standard.\n\n"
            f"Question: {question}"
        )
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=self._types.GenerateContentConfig(temperature=0.0),
        )
        return (response.text or "").strip()


class FakeReferenceJudge:
    """Deterministic reference judge for tests / --fake mode (no network)."""

    def __init__(self, response: str = "Gold reference answer.") -> None:
        self._response = response

    def generate(self, question: str, expected_spec: str | None) -> str:
        return self._response


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text())


def generate_reference_answers(
    questions: list[dict],
    judge: ReferenceJudge,
    cache_path: Path | None = None,
) -> dict[str, str]:
    """Produce (and cache) a reference answer for every *answerable* question.

    Already-cached references are reused so this is a one-time generation cost.
    Unanswerable questions are skipped — they have no gold answer because the
    system is expected to abstain rather than answer them.
    """
    references: dict[str, str] = {}
    if cache_path is not None and cache_path.exists():
        try:
            references = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            references = {}

    missing = [
        q for q in questions if q.get("answerable") and q["id"] not in references
    ]

    if missing:
        logger.info("Generating {} reference answer(s) via Gemini judge...", len(missing))
        for q in missing:
            references[q["id"]] = judge.generate(q["query"], q.get("expected_spec"))

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(references, indent=2))

    return references


def run_pipeline(
    questions: list[dict],
    retriever,
    generator,
    release: str,
) -> list[QAPair]:
    """Run retrieval + grounded generation for every question.

    `retriever.retrieve(query, release)` and
    `generator.answer(query, retrieved)` mirror the live API path. Contexts
    are the raw text of the retrieved chunks (what ragas judges against).
    """
    pairs: list[QAPair] = []
    for q in questions:
        retrieved = retriever.retrieve(q["query"], release)
        result = generator.answer(q["query"], retrieved)
        contexts = [r.chunk.text for r in retrieved]
        pairs.append(
            QAPair(
                id=q["id"],
                query=q["query"],
                answer=result.answer,
                contexts=contexts,
                abstained=result.abstained,
            )
        )
    return pairs


def build_ragas_records(
    qa_pairs: list[QAPair],
    references: dict[str, str],
) -> list[dict]:
    """Assemble the ragas evaluation table (pure; no ragas import).

    Only questions that the system *answered* (did not abstain) are included:
    faithfulness / context-precision judge answer quality, which is undefined
    for an abstention. The `reference` column carries the Gemini-written gold
    answer so `context_recall` can run.

    Returns a list of dicts with keys: question, answer, contexts, reference.
    """
    records: list[dict] = []
    for pair in qa_pairs:
        if pair.abstained:
            continue
        records.append(
            {
                "question": pair.query,
                "answer": pair.answer,
                "contexts": pair.contexts,
                "reference": references.get(pair.id, ""),
            }
        )
    return records


def build_evaluation_dataset(records: list[dict]):
    """Convert the assembled records into a ragas EvaluationDataset.

    Imported lazily so this module can be imported (and `build_ragas_records`
    unit-tested) even when ragas / datasets are not installed.
    """
    from datasets import Dataset
    from ragas import EvaluationDataset

    columns = {key: [r[key] for r in records] for key in ("question", "answer", "contexts", "reference")}
    hf_dataset = Dataset.from_dict(columns)
    return EvaluationDataset.from_hf_dataset(hf_dataset)


def build_gemini_judge_llm(api_key: str, model_name: str = "gemini-2.0-flash"):
    """Return a ragas-compatible LLM wired to Gemini (no OpenAI)."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The ragas Gemini judge requires `langchain-google-genai` and `ragas`. "
            "Install with: pip install ragas langchain-google-genai"
        ) from exc

    chat = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    return LangchainLLMWrapper(chat)


def evaluate_dataset(dataset_obj, judge_llm) -> dict:
    """Run ragas metrics (faithfulness, context_precision, context_recall)."""
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness

    result = evaluate(
        dataset_obj,
        metrics=[faithfulness, context_precision, context_recall],
        llm=judge_llm,
    )
    return result


def main() -> int:
    from app.dependencies import get_generator, get_retriever
    from app.logging_config import configure_logging

    parser = argparse.ArgumentParser(description="Ragas LLM-judge evaluation (Gemini, no OpenAI).")
    parser.add_argument("--fake", action="store_true", help="No network; dump assembled dataset only.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    configure_logging()

    dataset = load_dataset(args.dataset)
    release = dataset["release"]
    questions = dataset["questions"]
    if args.limit is not None:
        questions = questions[: args.limit]

    if args.fake:
        logger.info("Fake mode: assembling ragas dataset without network/LLM calls.")
        references: dict[str, str] = {
            q["id"]: "Gold reference answer." for q in questions if q.get("answerable")
        }
        qa_pairs = [
            QAPair(
                id=q["id"],
                query=q["query"],
                answer="The UE shall send a REGISTRATION REQUEST [E1].",
                contexts=["The UE shall send a REGISTRATION REQUEST message to the AMF."],
                abstained=False,
            )
            for q in questions
            if q.get("answerable")
        ]
        records = build_ragas_records(qa_pairs, references)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "ragas_dataset.json").write_text(json.dumps(records, indent=2))
        logger.info("Wrote {} ragas records to {}", len(records), RESULTS_DIR / "ragas_dataset.json")
        return 0

    import os

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY must be set in the environment to run the real ragas judge.")
        return 1

    judge = GeminiReferenceJudge(api_key)
    references = generate_reference_answers(questions, judge, REFERENCES_PATH)

    retriever = get_retriever()
    generator = get_generator()
    qa_pairs = run_pipeline(questions, retriever, generator, release)
    records = build_ragas_records(qa_pairs, references)

    if not records:
        logger.warning("No answered questions to evaluate (all abstained?).")
        return 0

    judge_llm = build_gemini_judge_llm(api_key)
    dataset_obj = build_evaluation_dataset(records)
    results = evaluate_dataset(dataset_obj, judge_llm)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "ragas_results.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info("Ragas results written to {}", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())