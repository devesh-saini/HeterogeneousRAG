from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QAPair:
    question_id: str
    question: str
    answer: str
    evidence: list[str]
    reference_answers: tuple[str, ...] = ()
    answer_type: str = "unknown"
    reference_answer_types: tuple[str, ...] = ()

    @property
    def answers(self) -> list[str]:
        """Return every acceptable answer while preserving legacy loaders."""
        references = [answer.strip() for answer in self.reference_answers if answer.strip()]
        if self.answer.strip() and self.answer.strip() not in references:
            references.insert(0, self.answer.strip())
        return references or [self.answer]

    @property
    def answer_types(self) -> list[str]:
        if len(self.reference_answer_types) == len(self.answers):
            return list(self.reference_answer_types)
        return [self.answer_type] * len(self.answers)


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


def reference_answers(value: object) -> tuple[str, ...]:
    """Extract alternate gold answers from common Hugging Face QA shapes."""
    if isinstance(value, dict):
        for key in ("text", "answers"):
            if key in value:
                return reference_answers(value[key])
        normalized = normalize_answer_value(value).strip()
        return (normalized,) if normalized else ()
    if isinstance(value, list):
        references: list[str] = []
        for item in value:
            if isinstance(item, dict):
                references.extend(reference_answers(item))
            else:
                normalized = str(item).strip()
                if normalized:
                    references.append(normalized)
        return tuple(dict.fromkeys(references))
    normalized = str(value).strip() if value not in (None, "") else ""
    return (normalized,) if normalized else ()


def infer_answer_type(answer: str) -> str:
    normalized = answer.strip().lower()
    if normalized in {"yes", "no", "true", "false"}:
        return "yes_no"
    if normalized in {"unanswerable", "insufficient evidence"}:
        return "unanswerable"
    return "free_form"
