from __future__ import annotations

import json
import time
from typing import Callable

from config.domains import RESULTS_DB_PATH
from data.evaluation.common import QAPair
from evaluation.metrics import retrieval_recall, score_answer
from evaluation.schema import EVALUATION_VERSION, RESULT_COLUMN_MIGRATIONS
from pipeline.graph import build_rag_graph
from pipeline.state import RAGState


CREATE_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id TEXT,
    domain TEXT,
    config_name TEXT,
    model_assignment TEXT,
    em_score REAL,
    f1_score REAL,
    answer_precision REAL,
    answer_recall REAL,
    reference_contained INTEGER,
    verbosity_ratio REAL,
    matched_reference_index INTEGER,
    reference_count INTEGER,
    retrieval_recall REAL,
    retry_count INTEGER,
    hallucination INTEGER,
    groundedness_pass INTEGER,
    relevance_pass INTEGER,
    completeness_pass INTEGER,
    verifier_parse_success INTEGER,
    correctness_pass INTEGER,
    failure_category TEXT,
    total_cost_usd REAL,
    total_latency_ms INTEGER,
    node_metadata TEXT,
    original_query TEXT,
    rewritten_query TEXT,
    gold_evidence TEXT,
    retrieved_chunks TEXT,
    retrieved_metadata TEXT,
    verification_result TEXT,
    synthesized_answer TEXT,
    gold_answer TEXT,
    gold_answers TEXT,
    answer_type TEXT,
    evaluation_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable


def _load_pairs(domain: str, limit: int | None) -> list[QAPair]:
    loaders: dict[str, Callable[[int | None], list[QAPair]]] = {
        "cs": __import__("data.evaluation.load_cs", fromlist=["load_qa_pairs"]).load_qa_pairs,
        "computerScience": __import__(
            "data.evaluation.load_cs", fromlist=["load_qa_pairs"]
        ).load_qa_pairs,
        "medical": __import__("data.evaluation.load_medical", fromlist=["load_qa_pairs"]).load_qa_pairs,
        "law": __import__("data.evaluation.load_law", fromlist=["load_qa_pairs"]).load_qa_pairs,
        "finance": __import__("data.evaluation.load_finance", fromlist=["load_qa_pairs"]).load_qa_pairs,
    }
    return loaders[domain](limit)


def _ensure_db():
    from sqlalchemy import create_engine, text

    RESULTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{RESULTS_DB_PATH}")
    with engine.begin() as conn:
        conn.execute(text(CREATE_RESULTS_TABLE))
        existing_columns = {
            row[1] for row in conn.execute(text("PRAGMA table_info(results)")).fetchall()
        }
        for column_name, column_type in RESULT_COLUMN_MIGRATIONS.items():
            if column_name not in existing_columns:
                conn.execute(
                    text(f"ALTER TABLE results ADD COLUMN {column_name} {column_type}")
                )
    return engine


def _metadata_totals(metadata: dict) -> tuple[float, int]:
    values = metadata.get("events", metadata.values())
    total_cost = sum(float(node.get("cost_usd", 0.0)) for node in values)
    total_latency = sum(int(node.get("latency_ms", 0)) for node in values)
    return total_cost, total_latency


def _failure_category(
    *,
    grounded: bool,
    relevant: bool,
    complete: bool,
    correctness_pass: bool,
    verifier_parse_success: bool,
) -> str:
    if not verifier_parse_success:
        return "verifier_parse_error"
    if not grounded:
        return "unsupported_claims"
    if not relevant:
        return "irrelevant_answer"
    if not complete:
        return "incomplete_answer"
    if not correctness_pass:
        return "low_reference_overlap"
    return "pass"


def run_evaluation(
    *,
    domain: str,
    config_name: str,
    model_assignment: dict[str, str],
    n_questions: int | None,
) -> None:
    graph = build_rag_graph()
    qa_pairs = _load_pairs(domain, n_questions)
    engine = _ensure_db()

    for pair in tqdm(qa_pairs, desc=f"{domain}/{config_name}"):
        state: RAGState = {
            "question_id": pair.question_id,
            "original_query": pair.question,
            "domain": domain,
            "config_name": config_name,
            "retry_count": 0,
            "gold_answer": pair.answer,
            "gold_answers": pair.answers,
            "gold_evidence": pair.evidence,
            "answer_type": pair.answer_type,
            "metadata": {},
            "model_assignment": model_assignment,
        }
        start = time.perf_counter()
        result = graph.invoke(state)
        wall_latency_ms = int((time.perf_counter() - start) * 1000)
        answer = result.get("synthesized_answer", "")
        answer_scores = score_answer(answer, pair.answers)
        em_score = float(answer_scores["exact_match"])
        f1_score = float(answer_scores["f1"])
        retrieved_chunks = result.get("retrieved_chunks", [])
        retrieved_metadata = result.get("retrieved_metadata", [])
        recall = retrieval_recall(retrieved_chunks, pair.evidence, retrieved_metadata)
        metadata = result.get("metadata", {})
        total_cost, node_latency_ms = _metadata_totals(metadata)
        total_latency_ms = max(wall_latency_ms, node_latency_ms)
        verification = result.get("verification_result", {})
        grounded = bool(verification.get("groundedness_verdict", False))
        relevant = bool(verification.get("relevance_verdict", False))
        complete = bool(verification.get("completeness_verdict", False))
        verifier_parse_success = bool(verification.get("parse_success", False))
        reference_coverage_pass = float(answer_scores["answer_recall"]) >= 0.8
        correctness_pass = bool(
            em_score
            or f1_score >= 0.5
            or answer_scores["reference_contained"]
            or (reference_coverage_pass and grounded and relevant)
        )
        # Hallucination now means unsupported content, independent of answer correctness.
        hallucination = int(not grounded) if verifier_parse_success else None
        failure_category = _failure_category(
            grounded=grounded,
            relevant=relevant,
            complete=complete,
            correctness_pass=correctness_pass,
            verifier_parse_success=verifier_parse_success,
        )

        with engine.begin() as conn:
            from sqlalchemy import text

            conn.execute(
                text(
                    """
                    INSERT INTO results (
                        question_id, domain, config_name, model_assignment,
                        em_score, f1_score, answer_precision, answer_recall,
                        reference_contained, verbosity_ratio, matched_reference_index,
                        reference_count, retrieval_recall, retry_count, hallucination,
                        groundedness_pass, relevance_pass, completeness_pass,
                        verifier_parse_success, correctness_pass, failure_category,
                        total_cost_usd, total_latency_ms, node_metadata,
                        original_query, rewritten_query, gold_evidence,
                        retrieved_chunks, retrieved_metadata, verification_result,
                        synthesized_answer, gold_answer, gold_answers, answer_type,
                        evaluation_version
                    )
                    VALUES (
                        :question_id, :domain, :config_name, :model_assignment,
                        :em_score, :f1_score, :answer_precision, :answer_recall,
                        :reference_contained, :verbosity_ratio, :matched_reference_index,
                        :reference_count, :retrieval_recall, :retry_count, :hallucination,
                        :groundedness_pass, :relevance_pass, :completeness_pass,
                        :verifier_parse_success, :correctness_pass, :failure_category,
                        :total_cost_usd, :total_latency_ms, :node_metadata,
                        :original_query, :rewritten_query, :gold_evidence,
                        :retrieved_chunks, :retrieved_metadata, :verification_result,
                        :synthesized_answer, :gold_answer, :gold_answers, :answer_type,
                        :evaluation_version
                    )
                    """
                ),
                {
                    "question_id": pair.question_id,
                    "domain": domain,
                    "config_name": config_name,
                    "model_assignment": json.dumps(model_assignment),
                    "em_score": em_score,
                    "f1_score": f1_score,
                    "answer_precision": answer_scores["answer_precision"],
                    "answer_recall": answer_scores["answer_recall"],
                    "reference_contained": int(answer_scores["reference_contained"]),
                    "verbosity_ratio": answer_scores["verbosity_ratio"],
                    "matched_reference_index": answer_scores["matched_reference_index"],
                    "reference_count": answer_scores["reference_count"],
                    "retrieval_recall": recall,
                    "retry_count": int(result.get("retry_count", 0)),
                    "hallucination": hallucination,
                    "groundedness_pass": int(grounded),
                    "relevance_pass": int(relevant),
                    "completeness_pass": int(complete),
                    "verifier_parse_success": int(verifier_parse_success),
                    "correctness_pass": int(correctness_pass),
                    "failure_category": failure_category,
                    "total_cost_usd": total_cost,
                    "total_latency_ms": total_latency_ms,
                    "node_metadata": json.dumps(metadata),
                    "original_query": pair.question,
                    "rewritten_query": result.get("rewritten_query", ""),
                    "gold_evidence": json.dumps(pair.evidence),
                    "retrieved_chunks": json.dumps(retrieved_chunks),
                    "retrieved_metadata": json.dumps(retrieved_metadata),
                    "verification_result": json.dumps(result.get("verification_result", {})),
                    "synthesized_answer": answer,
                    "gold_answer": pair.answer,
                    "gold_answers": json.dumps(pair.answers),
                    "answer_type": pair.answer_type,
                    "evaluation_version": EVALUATION_VERSION,
                },
            )
