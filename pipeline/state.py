from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    original_query: str
    domain: str
    config_name: str
    question_id: str

    rewritten_query: str
    retrieved_chunks: list[str]
    retrieved_metadata: list[dict[str, Any]]
    reranked_chunks: list[str]
    synthesized_answer: str
    verification_result: dict[str, Any]

    retry_count: int

    gold_answer: str
    gold_answers: list[str]
    gold_evidence: list[str]
    answer_type: str

    metadata: dict[str, Any]
    model_assignment: dict[str, str]
    failed: bool
