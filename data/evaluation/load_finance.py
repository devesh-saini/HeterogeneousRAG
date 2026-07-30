from __future__ import annotations

from datasets import load_dataset

from config.domains import get_domain
from data.evaluation.common import (
    QAPair,
    first_present,
    infer_answer_type,
    normalize_answer_value,
    reference_answers,
)


def _evidence_texts(value: object) -> list[str]:
    if isinstance(value, list):
        return [
            str(item.get("evidence_text", ""))
            if isinstance(item, dict)
            else str(item)
            for item in value
            if (item.get("evidence_text") if isinstance(item, dict) else item)
        ]
    return [str(value)] if value not in (None, "") else []


def load_qa_pairs(limit: int | None = None) -> list[QAPair]:
    dataset = load_dataset(get_domain("finance").dataset_id)
    split = dataset["test"] if "test" in dataset else dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    pairs: list[QAPair] = []
    for i, row in enumerate(split):
        question = str(first_present(row, ["question", "query"], ""))
        answer_value = first_present(row, ["answer", "answers"], "")
        answer = normalize_answer_value(answer_value)
        evidence = first_present(row, ["evidence", "context"], "")
        pairs.append(
            QAPair(
                str(first_present(row, ["financebench_id", "id", "question_id"], i)),
                question,
                answer,
                _evidence_texts(evidence),
                reference_answers(answer_value),
                infer_answer_type(answer),
            )
        )
        if limit is not None and len(pairs) >= limit:
            break
    return pairs
