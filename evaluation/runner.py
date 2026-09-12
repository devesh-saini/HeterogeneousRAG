from __future__ import annotations

import json
import time
from typing import Callable

from config.domains import RESULTS_DB_PATH
from data.evaluation.common import QAPair
from evaluation.metrics import retrieval_recall, score_answer
from evaluation.schema import (
    EVALUATION_VERSION,
    SEMANTIC_MODEL_ID,
    SCORING_VERSION,
    RESULT_COLUMN_MIGRATIONS,
)
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
    f2_score REAL,
    rouge_l_f1 REAL,
    semantic_similarity REAL,
    semantic_model TEXT,
    type_accuracy REAL,
    answer_precision REAL,
    answer_recall REAL,
    reference_contained INTEGER,
    verbosity_ratio REAL,
    matched_reference_index INTEGER,
    matched_answer_type TEXT,
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
    gold_answer_types TEXT,
    answer_type TEXT,
    evaluation_version TEXT,
    scoring_version TEXT,
    experiment_id TEXT,
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
        conn.execute(text("DROP INDEX IF EXISTS uq_results_experiment_config_question"))
        conn.execute(
            text(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    uq_results_experiment_domain_config_question
                ON results (experiment_id, domain, config_name, question_id)
                WHERE experiment_id IS NOT NULL
                """
            )
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
    experiment_id: str | None = None,
    resume: bool = False,
    semantic_scoring: bool = True,
) -> None:
    graph = build_rag_graph()
    qa_pairs = _load_pairs(domain, n_questions)
    engine = _ensure_db()

    if experiment_id or resume:
        from sqlalchemy import text

        query = """
            SELECT question_id FROM results
            WHERE domain = :domain
              AND config_name = :config_name
              AND evaluation_version = :evaluation_version
        """
        query_params = {
            "domain": domain,
            "config_name": config_name,
            "evaluation_version": EVALUATION_VERSION,
        }
        if experiment_id:
            query += " AND experiment_id = :experiment_id"
            query_params["experiment_id"] = experiment_id
        else:
            query += " AND experiment_id IS NULL"
        with engine.begin() as conn:
            existing_question_ids = {
                str(row[0])
                for row in conn.execute(
                    text(query),
                    query_params,
                )
            }
        overlap = existing_question_ids.intersection(pair.question_id for pair in qa_pairs)
        if experiment_id and overlap and not resume:
            raise ValueError(
                f"Experiment '{experiment_id}' already has {len(overlap)} {config_name} "
                "result(s). Use --resume to skip them or choose another experiment id."
            )
        if resume:
            qa_pairs = [
                pair for pair in qa_pairs if pair.question_id not in existing_question_ids
            ]

    semantic_encoder = None

    for pair in tqdm(qa_pairs, desc=f"{domain}/{config_name}"):
        state: RAGState = {
            "question_id": pair.question_id,
            "original_query": pair.question,
            "domain": domain,
            "config_name": config_name,
            "retry_count": 0,
            "gold_answer": pair.answer,
            "gold_answers": pair.answers,
            "gold_answer_types": pair.answer_types,
            "gold_evidence": pair.evidence,
            "answer_type": pair.answer_type,
            "metadata": {},
            "model_assignment": model_assignment,
        }
        start = time.perf_counter()
        result = graph.invoke(state)
        wall_latency_ms = int((time.perf_counter() - start) * 1000)
        answer = result.get("synthesized_answer", "")
        if semantic_scoring and semantic_encoder is None:
            # Retrieval has already loaded this model, so semantic scoring reuses
            # the same instance instead of consuming more memory.
            from pipeline.nodes.retriever import get_embedder

            semantic_encoder = get_embedder()
        answer_scores = score_answer(
            answer,
            pair.answers,
            answer_type=pair.answer_type,
            reference_types=pair.answer_types,
            semantic_encoder=semantic_encoder,
        )
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
        type_accuracy = answer_scores["type_accuracy"]
        correctness_pass = bool(
            type_accuracy == 1.0
            or em_score
            or f1_score >= 0.5
            or answer_scores["reference_contained"]
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
                        em_score, f1_score, f2_score, rouge_l_f1,
                        semantic_similarity, semantic_model, type_accuracy,
                        answer_precision, answer_recall,
                        reference_contained, verbosity_ratio, matched_reference_index,
                        matched_answer_type, reference_count,
                        retrieval_recall, retry_count, hallucination,
                        groundedness_pass, relevance_pass, completeness_pass,
                        verifier_parse_success, correctness_pass, failure_category,
                        total_cost_usd, total_latency_ms, node_metadata,
                        original_query, rewritten_query, gold_evidence,
                        retrieved_chunks, retrieved_metadata, verification_result,
                        synthesized_answer, gold_answer, gold_answers,
                        gold_answer_types, answer_type,
                        evaluation_version, scoring_version, experiment_id
                    )
                    VALUES (
                        :question_id, :domain, :config_name, :model_assignment,
                        :em_score, :f1_score, :f2_score, :rouge_l_f1,
                        :semantic_similarity, :semantic_model, :type_accuracy,
                        :answer_precision, :answer_recall,
                        :reference_contained, :verbosity_ratio, :matched_reference_index,
                        :matched_answer_type, :reference_count,
                        :retrieval_recall, :retry_count, :hallucination,
                        :groundedness_pass, :relevance_pass, :completeness_pass,
                        :verifier_parse_success, :correctness_pass, :failure_category,
                        :total_cost_usd, :total_latency_ms, :node_metadata,
                        :original_query, :rewritten_query, :gold_evidence,
                        :retrieved_chunks, :retrieved_metadata, :verification_result,
                        :synthesized_answer, :gold_answer, :gold_answers,
                        :gold_answer_types, :answer_type,
                        :evaluation_version, :scoring_version, :experiment_id
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
                    "f2_score": answer_scores["f2"],
                    "rouge_l_f1": answer_scores["rouge_l_f1"],
                    "semantic_similarity": answer_scores["semantic_similarity"],
                    "semantic_model": (
                        SEMANTIC_MODEL_ID if answer_scores["semantic_similarity"] is not None else None
                    ),
                    "type_accuracy": type_accuracy,
                    "answer_precision": answer_scores["answer_precision"],
                    "answer_recall": answer_scores["answer_recall"],
                    "reference_contained": int(answer_scores["reference_contained"]),
                    "verbosity_ratio": answer_scores["verbosity_ratio"],
                    "matched_reference_index": answer_scores["matched_reference_index"],
                    "matched_answer_type": answer_scores["matched_answer_type"],
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
                    "gold_answer_types": json.dumps(pair.answer_types),
                    "answer_type": pair.answer_type,
                    "evaluation_version": EVALUATION_VERSION,
                    "scoring_version": SCORING_VERSION,
                    "experiment_id": experiment_id,
                },
            )
    engine.dispose()
