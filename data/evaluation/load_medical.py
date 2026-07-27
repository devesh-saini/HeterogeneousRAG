from __future__ import annotations

import ast

from datasets import load_dataset

from config.domains import get_domain
from data.evaluation.common import QAPair, first_present, normalize_answer_value


def _normalize_evidence_ids(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [item.strip() for item in value.strip("[]").split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value not in (None, "") else []


def load_qa_pairs(limit: int | None = None) -> list[QAPair]:
    dataset = load_dataset(get_domain("medical").dataset_id, "question-answer-passages")
    split = dataset["test"] if "test" in dataset else next(iter(dataset.values()))
    pairs: list[QAPair] = []
    for i, row in enumerate(split):
        question = str(first_present(row, ["question", "query"], ""))
        answer = normalize_answer_value(first_present(row, ["answer", "answers"], ""))
        evidence = first_present(row, ["relevant_passage_ids", "evidence", "passages"], [])
        pairs.append(
            QAPair(
                str(first_present(row, ["id", "question_id"], i)),
                question,
                answer,
                _normalize_evidence_ids(evidence),
            )
        )
        if limit is not None and len(pairs) >= limit:
            break
    return pairs
