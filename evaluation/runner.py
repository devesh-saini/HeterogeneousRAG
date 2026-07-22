from __future__ import annotations

import json
import time
from typing import Callable

from config.domains import RESULTS_DB_PATH
from data.evaluation.common import QAPair
from evaluation.metrics import exact_match, retrieval_recall, token_f1
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
    retrieval_recall REAL,
    retry_count INTEGER,
    hallucination INTEGER,
    total_cost_usd REAL,
    total_latency_ms INTEGER,
    node_metadata TEXT,
    synthesized_answer TEXT,
    gold_answer TEXT,
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
    return engine


def _metadata_totals(metadata: dict) -> tuple[float, int]:
    total_cost = sum(float(node.get("cost_usd", 0.0)) for node in metadata.values())
    total_latency = sum(int(node.get("latency_ms", 0)) for node in metadata.values())
    return total_cost, total_latency


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
            "gold_evidence": pair.evidence,
            "metadata": {},
            "model_assignment": model_assignment,
        }
        start = time.perf_counter()
        result = graph.invoke(state)
        wall_latency_ms = int((time.perf_counter() - start) * 1000)
        answer = result.get("synthesized_answer", "")
        em_score = exact_match(answer, pair.answer)
        f1_score = token_f1(answer, pair.answer)
        recall = retrieval_recall(result.get("retrieved_chunks", []), pair.evidence)
        metadata = result.get("metadata", {})
        total_cost, node_latency_ms = _metadata_totals(metadata)
        total_latency_ms = max(wall_latency_ms, node_latency_ms)
        verifier_flagged = not result.get("verification_result", {}).get("verdict", False)
        hallucination = int(verifier_flagged and em_score == 0.0 and f1_score < 0.5)

        with engine.begin() as conn:
            from sqlalchemy import text

            conn.execute(
                text(
                    """
                    INSERT INTO results (
                        question_id, domain, config_name, model_assignment,
                        em_score, f1_score, retrieval_recall, retry_count, hallucination,
                        total_cost_usd, total_latency_ms, node_metadata,
                        synthesized_answer, gold_answer
                    )
                    VALUES (
                        :question_id, :domain, :config_name, :model_assignment,
                        :em_score, :f1_score, :retrieval_recall, :retry_count, :hallucination,
                        :total_cost_usd, :total_latency_ms, :node_metadata,
                        :synthesized_answer, :gold_answer
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
                    "retrieval_recall": recall,
                    "retry_count": int(result.get("retry_count", 0)),
                    "hallucination": hallucination,
                    "total_cost_usd": total_cost,
                    "total_latency_ms": total_latency_ms,
                    "node_metadata": json.dumps(metadata),
                    "synthesized_answer": answer,
                    "gold_answer": pair.answer,
                },
            )
