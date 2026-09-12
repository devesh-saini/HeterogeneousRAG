from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from config.domains import RESULTS_DB_PATH
from evaluation.metrics import score_answer
from evaluation.schema import (
    EVALUATION_VERSION,
    SEMANTIC_MODEL_ID,
    RESULT_COLUMN_MIGRATIONS,
    SCORING_VERSION,
)


def _json_list(value: object, fallback: str) -> list[str]:
    try:
        parsed = json.loads(str(value)) if value else []
    except json.JSONDecodeError:
        parsed = []
    if isinstance(parsed, list):
        references = [str(item) for item in parsed if str(item).strip()]
        if references:
            return references
    return [fallback] if fallback.strip() else []


def _qasper_pairs() -> dict[str, Any]:
    from data.evaluation.load_cs import load_qa_pairs

    return {pair.question_id: pair for pair in load_qa_pairs(None)}


def _failure_category(row: sqlite3.Row, correctness_pass: bool) -> str:
    if not bool(row["verifier_parse_success"]):
        return "verifier_parse_error"
    if not bool(row["groundedness_pass"]):
        return "unsupported_claims"
    if not bool(row["relevance_pass"]):
        return "irrelevant_answer"
    if not bool(row["completeness_pass"]):
        return "incomplete_answer"
    if not correctness_pass:
        return "low_reference_overlap"
    return "pass"


def _ensure_columns(connection: sqlite3.Connection) -> None:
    existing = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(results)")
    }
    for column_name, column_type in RESULT_COLUMN_MIGRATIONS.items():
        if column_name not in existing:
            connection.execute(
                f"ALTER TABLE results ADD COLUMN {column_name} {column_type}"
            )
    connection.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute answer metrics from stored outputs; no LLM calls are made."
    )
    parser.add_argument("--db", type=Path, default=RESULTS_DB_PATH)
    parser.add_argument("--evaluation-version", default=EVALUATION_VERSION)
    parser.add_argument("--config", choices=("homogeneous", "heterogeneous"))
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip MiniLM semantic similarity (all lexical metrics still run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    where = ["evaluation_version = ?"]
    params: list[Any] = [args.evaluation_version]
    if args.config:
        where.append("config_name = ?")
        params.append(args.config)

    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        _ensure_columns(connection)
        rows = connection.execute(
            f"SELECT * FROM results WHERE {' AND '.join(where)} ORDER BY id",
            params,
        ).fetchall()

        qasper_pairs: dict[str, Any] = {}
        if any(str(row["domain"]) in {"cs", "computerScience"} for row in rows):
            qasper_pairs = _qasper_pairs()

        semantic_encoder = None
        if not args.no_semantic and rows:
            try:
                from pipeline.nodes.retriever import get_embedder

                semantic_encoder = get_embedder()
            except Exception as exc:
                print(
                    f"Semantic model unavailable ({exc}); continuing with lexical metrics."
                )

        for row in rows:
            prediction = str(row["synthesized_answer"] or "")
            pair = qasper_pairs.get(str(row["question_id"]))
            if pair is not None:
                references = pair.answers
                reference_types = pair.answer_types
                answer_type = pair.answer_type
            else:
                references = _json_list(
                    row["gold_answers"], str(row["gold_answer"] or "")
                )
                answer_type = str(row["answer_type"] or "unknown")
                reference_types = _json_list(
                    row["gold_answer_types"], answer_type
                )
            scores = score_answer(
                prediction,
                references,
                answer_type=answer_type,
                reference_types=reference_types,
                semantic_encoder=semantic_encoder,
            )
            type_accuracy = scores["type_accuracy"]
            correctness_pass = bool(
                type_accuracy == 1.0
                or scores["exact_match"]
                or float(scores["f1"]) >= 0.5
                or scores["reference_contained"]
            )
            connection.execute(
                """
                UPDATE results
                SET em_score = ?, f1_score = ?, f2_score = ?, rouge_l_f1 = ?,
                    semantic_similarity = ?, semantic_model = ?, type_accuracy = ?,
                    answer_precision = ?, answer_recall = ?,
                    reference_contained = ?, verbosity_ratio = ?,
                    matched_reference_index = ?, reference_count = ?,
                    matched_answer_type = ?, correctness_pass = ?,
                    failure_category = ?, scoring_version = ?,
                    gold_answer = ?, gold_answers = ?, gold_answer_types = ?,
                    answer_type = ?
                WHERE id = ?
                """,
                (
                    scores["exact_match"],
                    scores["f1"],
                    scores["f2"],
                    scores["rouge_l_f1"],
                    scores["semantic_similarity"],
                    SEMANTIC_MODEL_ID if scores["semantic_similarity"] is not None else None,
                    scores["type_accuracy"],
                    scores["answer_precision"],
                    scores["answer_recall"],
                    int(scores["reference_contained"]),
                    scores["verbosity_ratio"],
                    scores["matched_reference_index"],
                    scores["reference_count"],
                    scores["matched_answer_type"],
                    int(correctness_pass),
                    _failure_category(row, correctness_pass),
                    SCORING_VERSION,
                    references[0] if references else "",
                    json.dumps(references),
                    json.dumps(reference_types),
                    answer_type,
                    row["id"],
                ),
            )
        connection.commit()

    semantic_status = "with semantic similarity" if semantic_encoder else "lexical only"
    print(f"Rescored {len(rows)} rows using {SCORING_VERSION} ({semantic_status}).")


if __name__ == "__main__":
    main()
