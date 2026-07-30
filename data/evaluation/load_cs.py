from __future__ import annotations

from data.evaluation.common import QAPair, normalize_answer_value
from data.ingestion.sources import load_qasper_documents


def _answers(answers: object) -> list[dict]:
    normalized: list[dict] = []
    if not isinstance(answers, list):
        return normalized
    for annotation in answers:
        if not isinstance(annotation, dict):
            continue
        answer = annotation.get("answer", annotation)
        if isinstance(answer, dict):
            normalized.append(answer)
    return normalized


def _answer_text(answer: dict) -> str:
    if answer.get("unanswerable"):
        return "unanswerable"
    free_form = str(answer.get("free_form_answer") or "").strip()
    if free_form:
        return free_form
    extractive_spans = answer.get("extractive_spans")
    if isinstance(extractive_spans, list) and extractive_spans:
        return normalize_answer_value(extractive_spans)
    yes_no = answer.get("yes_no")
    if yes_no is not None:
        return str(yes_no)
    return normalize_answer_value(answer)


def _answer_type(answer: dict) -> str:
    if answer.get("unanswerable"):
        return "unanswerable"
    if str(answer.get("free_form_answer") or "").strip():
        return "free_form"
    if answer.get("extractive_spans"):
        return "extractive"
    if answer.get("yes_no") is not None:
        return "yes_no"
    return "unknown"


def load_qa_pairs(limit: int | None = None) -> list[QAPair]:
    pairs: list[QAPair] = []
    for paper in load_qasper_documents("validation"):
        for qa in paper.get("qas", []):
            if not isinstance(qa, dict):
                continue
            annotations = _answers(qa.get("answers"))
            if not annotations:
                continue
            reference_answers = tuple(
                dict.fromkeys(
                    text for text in (_answer_text(answer) for answer in annotations) if text.strip()
                )
            )
            evidence = list(
                dict.fromkeys(
                    str(item)
                    for answer in annotations
                    for item in (
                        answer.get("highlighted_evidence")
                        or answer.get("evidence")
                        or []
                    )
                    if str(item).strip()
                )
            )
            question = str(qa.get("question", ""))
            pair = QAPair(
                str(qa.get("question_id", len(pairs))),
                question,
                reference_answers[0] if reference_answers else "",
                evidence,
                reference_answers,
                _answer_type(annotations[0]),
            )
            if pair.question and pair.answer:
                pairs.append(pair)
            if limit is not None and len(pairs) >= limit:
                return pairs
    return pairs
