from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QAPair:
    question_id: str
    question: str
    answer: str
    evidence: list[str]


def first_present(row: dict, names: list[str], default: object = "") -> object:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def normalize_answer_value(value: object) -> str:
    if isinstance(value, list):
        return "; ".join(map(str, value))
    if isinstance(value, dict):
        for key in ("answer", "free_form_answer", "extractive_spans"):
            if key in value:
                return normalize_answer_value(value[key])
    return str(value)
