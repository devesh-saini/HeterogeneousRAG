from __future__ import annotations

from datasets import load_dataset

from config.domains import get_domain
from data.evaluation.common import QAPair, first_present, reference_answers


def load_qa_pairs(limit: int | None = None) -> list[QAPair]:
    dataset = load_dataset(get_domain("law").dataset_id)
    split = dataset["test"] if "test" in dataset else dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    pairs: list[QAPair] = []
    for i, row in enumerate(split):
        question = str(first_present(row, ["question", "query"], ""))
        answers = reference_answers(first_present(row, ["answer", "answers"], ""))
        answer = answers[0] if answers else ""
        evidence = first_present(row, ["context", "evidence"], "")
        pairs.append(
            QAPair(
                str(first_present(row, ["id", "question_id"], i)),
                question,
                answer,
                [str(evidence)],
                answers,
                "extractive",
            )
        )
        if limit is not None and len(pairs) >= limit:
            break
    return pairs
