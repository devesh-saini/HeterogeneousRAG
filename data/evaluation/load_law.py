from __future__ import annotations

from data.evaluation.common import QAPair
from data.ingestion.sources import load_cuad_documents


def load_qa_pairs(limit: int | None = None) -> list[QAPair]:
    pairs: list[QAPair] = []
    for document in load_cuad_documents("test").get("data", []):
        if not isinstance(document, dict):
            continue
        source_id = str(document.get("title", ""))
        for paragraph in document.get("paragraphs", []):
            if not isinstance(paragraph, dict):
                continue
            for qa in paragraph.get("qas", []):
                if not isinstance(qa, dict):
                    continue
                answers = tuple(
                    dict.fromkeys(
                        str(answer.get("text", "")).strip()
                        for answer in qa.get("answers", [])
                        if isinstance(answer, dict) and str(answer.get("text", "")).strip()
                    )
                )
                if not answers:
                    answers = ("unanswerable",)
                pairs.append(
                    QAPair(
                        str(qa.get("id", len(pairs))),
                        str(qa.get("question", "")),
                        answers[0],
                        list(answers) if answers != ("unanswerable",) else [],
                        answers,
                        "extractive" if answers != ("unanswerable",) else "unanswerable",
                        (),
                        source_id or None,
                    )
                )
                if limit is not None and len(pairs) >= limit:
                    return pairs
    return pairs
