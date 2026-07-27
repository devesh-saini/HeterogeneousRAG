from __future__ import annotations

from data.evaluation.common import QAPair, normalize_answer_value
from data.ingestion.sources import load_qasper_documents


def _first_answer(answers: object) -> dict:
    if isinstance(answers, list) and answers:
        first = answers[0]
        if isinstance(first, dict):
            answer = first.get("answer", first)
            return answer if isinstance(answer, dict) else {"answer": answer}
    return {}


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


def load_qa_pairs(limit: int | None = None) -> list[QAPair]:
    pairs: list[QAPair] = []
    for paper in load_qasper_documents("validation"):
        for qa in paper.get("qas", []):
            if not isinstance(qa, dict):
                continue
            answer = _first_answer(qa.get("answers"))
            evidence = answer.get("highlighted_evidence") or answer.get("evidence") or []
            question = str(qa.get("question", ""))
            pair = QAPair(
                str(qa.get("question_id", len(pairs))),
                question,
                _answer_text(answer),
                list(evidence) if isinstance(evidence, list) else [str(evidence)],
            )
            if pair.question and pair.answer:
                pairs.append(pair)
            if limit is not None and len(pairs) >= limit:
                return pairs
    return pairs
