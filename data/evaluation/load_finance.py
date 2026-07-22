from __future__ import annotations

from datasets import load_dataset

from config.domains import get_domain
from data.evaluation.common import QAPair, first_present, normalize_answer_value


def load_qa_pairs(limit: int | None = None) -> list[QAPair]:
    dataset = load_dataset(get_domain("finance").dataset_id)
    split = dataset["test"] if "test" in dataset else dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    pairs: list[QAPair] = []
    for i, row in enumerate(split):
        question = str(first_present(row, ["question", "query"], ""))
        answer = normalize_answer_value(first_present(row, ["answer", "answers"], ""))
        evidence = first_present(row, ["evidence", "context"], "")
        pairs.append(QAPair(str(first_present(row, ["id", "question_id"], i)), question, answer, [str(evidence)]))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs
