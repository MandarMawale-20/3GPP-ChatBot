from __future__ import annotations

from evaluation.ragas_eval import (
    FakeReferenceJudge,
    QAPair,
    build_ragas_records,
    generate_reference_answers,
    load_dataset,
)

DATASET_PATH = __import__("pathlib").Path(__file__).resolve().parent.parent / "evaluation" / "dataset.json"


def _answerable_questions() -> list[dict]:
    return [q for q in load_dataset(DATASET_PATH)["questions"] if q.get("answerable")]


def test_load_dataset_has_questions_and_release() -> None:
    dataset = load_dataset(DATASET_PATH)
    assert dataset["release"] == "Rel-18"
    assert len(dataset["questions"]) > 0


def test_generate_reference_answers_only_for_answerable(tmp_path) -> None:
    questions = load_dataset(DATASET_PATH)["questions"]
    judge = FakeReferenceJudge(response="GOLD")
    cache = tmp_path / "refs.json"

    references = generate_reference_answers(questions, judge, cache_path=cache)

    # Every answerable question has a reference; unanswerable ones do not.
    for q in questions:
        if q.get("answerable"):
            assert references[q["id"]] == "GOLD"
        else:
            assert q["id"] not in references

    # Cache file was written.
    assert cache.exists()


def test_generate_reference_answers_reuses_cache(tmp_path) -> None:
    questions = _answerable_questions()
    assert questions, "need at least one answerable question to test cache reuse"

    # Seed the cache with references for every real answerable question id,
    # so none should be regenerated.
    seeded = {q["id"]: "PRE-EXISTING" for q in questions}
    cache = tmp_path / "refs.json"
    cache.write_text(__import__("json").dumps(seeded))

    # Spy judge that would raise if actually called for a missing question.
    class FailingJudge:
        def generate(self, question: str, expected_spec: str | None) -> str:
            raise AssertionError("should not be called when cache is present")

    references = generate_reference_answers(questions, FailingJudge(), cache_path=cache)  # type: ignore[arg-type]
    assert references == seeded


def test_build_ragas_records_excludes_abstentions() -> None:
    references = {"a": "gold-a"}
    qa_pairs = [
        QAPair(id="a", query="q", answer="ans", contexts=["ctx"], abstained=False),
        QAPair(id="b", query="q2", answer="ABSTAIN", contexts=[], abstained=True),
    ]

    records = build_ragas_records(qa_pairs, references)

    assert len(records) == 1
    record = records[0]
    assert record["question"] == "q"
    assert record["answer"] == "ans"
    assert record["contexts"] == ["ctx"]
    assert record["reference"] == "gold-a"


def test_build_ragas_records_uses_reference_when_present() -> None:
    qa_pairs = [
        QAPair(id="a", query="q", answer="ans", contexts=["ctx"], abstained=False),
    ]
    records = build_ragas_records(qa_pairs, {})
    assert records[0]["reference"] == ""  # no reference cached -> empty string