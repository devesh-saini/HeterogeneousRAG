from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from config.domains import RESULTS_DB_PATH
from evaluation.schema import RESULT_COLUMN_MIGRATIONS


SUMMARY_QUERY = """
SELECT
    domain,
    config_name,
    COALESCE(evaluation_version, 'legacy') AS evaluation_version,
    COALESCE(scoring_version, 'legacy') AS scoring_version,
    COALESCE(experiment_id, 'untracked') AS experiment_id,
    CASE
        WHEN json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.reranker')
         AND json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.synthesizer')
         AND json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.verifier')
        THEN json_extract(model_assignment, '$.rewriter')
        ELSE model_assignment
    END AS models,
    COUNT(*) AS runs,
    ROUND(AVG(em_score), 4) AS avg_em,
    ROUND(AVG(f1_score), 4) AS avg_f1,
    ROUND(AVG(f2_score), 4) AS avg_f2,
    ROUND(AVG(rouge_l_f1), 4) AS avg_rouge_l,
    ROUND(AVG(semantic_similarity), 4) AS avg_semantic,
    ROUND(AVG(type_accuracy), 4) AS categorical_accuracy,
    ROUND(AVG(answer_precision), 4) AS avg_answer_precision,
    ROUND(AVG(answer_recall), 4) AS avg_answer_recall,
    ROUND(AVG(reference_contained), 4) AS reference_containment_rate,
    ROUND(AVG(verbosity_ratio), 2) AS avg_verbosity_ratio,
    ROUND(AVG(correctness_pass), 4) AS correctness_rate,
    ROUND(MIN(f1_score), 4) AS min_f1,
    ROUND(MAX(f1_score), 4) AS max_f1,
    ROUND(AVG(CASE WHEN f1_score >= 0.5 THEN 1.0 ELSE 0.0 END), 4) AS f1_ge_0_5_rate,
    ROUND(AVG(retrieval_recall), 4) AS avg_retrieval_recall,
    ROUND(MAX(retrieval_recall), 4) AS max_retrieval_recall,
    ROUND(AVG(CASE WHEN retrieval_recall > 0 THEN 1.0 ELSE 0.0 END), 4) AS retrieval_hit_rate,
    ROUND(AVG(retry_count), 2) AS avg_retries,
    SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) AS retried_runs,
    MAX(retry_count) AS max_retries,
    ROUND(AVG(groundedness_pass), 4) AS groundedness_rate,
    ROUND(AVG(relevance_pass), 4) AS relevance_rate,
    ROUND(AVG(completeness_pass), 4) AS completeness_rate,
    ROUND(AVG(verifier_parse_success), 4) AS verifier_parse_rate,
    ROUND(AVG(hallucination), 4) AS unsupported_claim_rate,
    SUM(hallucination) AS hallucinations,
    ROUND(AVG(total_cost_usd), 6) AS avg_cost_usd,
    ROUND(SUM(total_cost_usd), 6) AS total_cost_usd,
    ROUND(AVG(total_latency_ms), 2) AS avg_latency_ms,
    MIN(total_latency_ms) AS min_latency_ms,
    MAX(total_latency_ms) AS max_latency_ms,
    ROUND(AVG(COALESCE(json_extract(node_metadata, '$.latest.rewriter.input_tokens'), json_extract(node_metadata, '$.rewriter.input_tokens'))), 1) AS avg_rewriter_in,
    ROUND(AVG(COALESCE(json_extract(node_metadata, '$.latest.reranker.input_tokens'), json_extract(node_metadata, '$.reranker.input_tokens'))), 1) AS avg_reranker_in,
    ROUND(AVG(COALESCE(json_extract(node_metadata, '$.latest.synthesizer.input_tokens'), json_extract(node_metadata, '$.synthesizer.input_tokens'))), 1) AS avg_synthesizer_in,
    ROUND(AVG(COALESCE(json_extract(node_metadata, '$.latest.verifier.input_tokens'), json_extract(node_metadata, '$.verifier.input_tokens'))), 1) AS avg_verifier_in
FROM analysis_results
GROUP BY domain, config_name, model_assignment,
         COALESCE(evaluation_version, 'legacy'),
         COALESCE(scoring_version, 'legacy'),
         COALESCE(experiment_id, 'untracked')
ORDER BY domain, config_name, models
"""

COMPACT_QUERY = """
SELECT
    domain,
    config_name,
    COALESCE(evaluation_version, 'legacy') AS evaluation_version,
    COALESCE(scoring_version, 'legacy') AS scoring_version,
    COALESCE(experiment_id, 'untracked') AS experiment_id,
    CASE
        WHEN json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.reranker')
         AND json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.synthesizer')
         AND json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.verifier')
        THEN json_extract(model_assignment, '$.rewriter')
        ELSE model_assignment
    END AS models,
    COUNT(*) AS runs,
    ROUND(AVG(f1_score), 4) AS avg_f1,
    ROUND(AVG(f2_score), 4) AS avg_f2,
    ROUND(AVG(rouge_l_f1), 4) AS avg_rouge_l,
    ROUND(AVG(semantic_similarity), 4) AS avg_semantic,
    ROUND(AVG(answer_precision), 4) AS avg_precision,
    ROUND(AVG(answer_recall), 4) AS avg_recall,
    ROUND(AVG(verbosity_ratio), 2) AS avg_verbosity,
    ROUND(AVG(correctness_pass), 4) AS correctness_rate,
    ROUND(AVG(retrieval_recall), 4) AS avg_retrieval,
    ROUND(AVG(CASE WHEN retrieval_recall > 0 THEN 1.0 ELSE 0.0 END), 4) AS retrieval_hit_rate,
    ROUND(AVG(retry_count), 2) AS avg_retries,
    ROUND(AVG(groundedness_pass), 4) AS groundedness_rate,
    ROUND(AVG(relevance_pass), 4) AS relevance_rate,
    ROUND(AVG(completeness_pass), 4) AS completeness_rate,
    ROUND(AVG(verifier_parse_success), 4) AS verifier_parse_rate,
    ROUND(AVG(hallucination), 4) AS unsupported_claim_rate,
    ROUND(AVG(total_latency_ms), 2) AS avg_latency_ms
FROM analysis_results
GROUP BY domain, config_name, model_assignment,
         COALESCE(evaluation_version, 'legacy'),
         COALESCE(scoring_version, 'legacy'),
         COALESCE(experiment_id, 'untracked')
ORDER BY domain, avg_f1 DESC
"""

FAILURE_QUERY = """
SELECT
    domain,
    config_name,
    COALESCE(evaluation_version, 'legacy') AS evaluation_version,
    COALESCE(scoring_version, 'legacy') AS scoring_version,
    COALESCE(experiment_id, 'untracked') AS experiment_id,
    CASE
        WHEN json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.reranker')
         AND json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.synthesizer')
         AND json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.verifier')
        THEN json_extract(model_assignment, '$.rewriter')
        ELSE model_assignment
    END AS models,
    COUNT(*) AS runs,
    SUM(CASE WHEN f1_score = 0 THEN 1 ELSE 0 END) AS zero_f1,
    SUM(CASE WHEN retrieval_recall = 0 THEN 1 ELSE 0 END) AS zero_retrieval,
    SUM(CASE WHEN retry_count >= 2 THEN 1 ELSE 0 END) AS max_retry_failures,
    SUM(CASE WHEN groundedness_pass = 0 THEN 1 ELSE 0 END) AS unsupported_claim_failures,
    SUM(CASE WHEN relevance_pass = 0 THEN 1 ELSE 0 END) AS relevance_failures,
    SUM(CASE WHEN completeness_pass = 0 THEN 1 ELSE 0 END) AS completeness_failures,
    SUM(CASE WHEN verifier_parse_success = 0 THEN 1 ELSE 0 END) AS verifier_parse_failures,
    SUM(CASE WHEN correctness_pass = 0 THEN 1 ELSE 0 END) AS correctness_failures
FROM analysis_results
GROUP BY domain, config_name, model_assignment,
         COALESCE(evaluation_version, 'legacy'),
         COALESCE(scoring_version, 'legacy'),
         COALESCE(experiment_id, 'untracked')
ORDER BY domain, config_name, models
"""

RECENT_QUERY = """
SELECT
    id,
    domain,
    config_name,
    COALESCE(evaluation_version, 'legacy') AS evaluation_version,
    COALESCE(scoring_version, 'legacy') AS scoring_version,
    COALESCE(experiment_id, 'untracked') AS experiment_id,
    CASE
        WHEN json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.reranker')
         AND json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.synthesizer')
         AND json_extract(model_assignment, '$.rewriter') = json_extract(model_assignment, '$.verifier')
        THEN json_extract(model_assignment, '$.rewriter')
        ELSE model_assignment
    END AS models,
    question_id,
    ROUND(em_score, 4) AS em,
    ROUND(f1_score, 4) AS f1,
    ROUND(f2_score, 4) AS f2,
    ROUND(rouge_l_f1, 4) AS rouge_l,
    ROUND(semantic_similarity, 4) AS semantic,
    ROUND(answer_precision, 4) AS precision,
    ROUND(answer_recall, 4) AS answer_recall,
    ROUND(verbosity_ratio, 2) AS verbosity,
    correctness_pass,
    groundedness_pass,
    relevance_pass,
    completeness_pass,
    verifier_parse_success,
    ROUND(retrieval_recall, 4) AS retrieval_recall,
    retry_count,
    hallucination AS unsupported_claims,
    total_latency_ms,
    created_at
FROM results
ORDER BY id DESC
LIMIT ?
"""


def _print_table(headers: list[str], rows: list[tuple[object, ...]]) -> None:
    if not rows:
        print("No rows found.")
        return

    values = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in values)) for i, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in values:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def _fetch(conn: sqlite3.Connection, query: str, params: tuple[object, ...] = ()) -> tuple[list[str], list[tuple[object, ...]]]:
    cursor = conn.execute(query, params)
    headers = [description[0] for description in cursor.description or []]
    return headers, cursor.fetchall()


def _ensure_analysis_columns(conn: sqlite3.Connection) -> None:
    """Make legacy databases queryable without rewriting historical rows."""
    existing_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(results)").fetchall()
    }
    for column_name, column_type in RESULT_COLUMN_MIGRATIONS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE results ADD COLUMN {column_name} {column_type}")
    conn.commit()


def _prepare_analysis_view(
    conn: sqlite3.Connection, *, include_duplicates: bool
) -> None:
    conn.execute("DROP VIEW IF EXISTS analysis_results")
    if include_duplicates:
        conn.execute("CREATE TEMP VIEW analysis_results AS SELECT * FROM results")
        return
    conn.execute(
        """
        CREATE TEMP VIEW analysis_results AS
        SELECT * FROM (
            SELECT results.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY domain, config_name, model_assignment,
                                    COALESCE(evaluation_version, 'legacy'),
                                    COALESCE(scoring_version, 'legacy'),
                                    COALESCE(experiment_id, 'untracked'),
                                    question_id
                       ORDER BY id DESC
                   ) AS analysis_row_number
            FROM results
        )
        WHERE analysis_row_number = 1
        """
    )


def _json_loads(value: object, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _print_recommendations(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT domain, config_name, model_assignment,
               AVG(f1_score) AS avg_f1,
               AVG(retrieval_recall) AS avg_retrieval,
               AVG(hallucination) AS hallucination_rate,
               AVG(total_latency_ms) AS avg_latency
        FROM analysis_results
        GROUP BY domain, config_name, model_assignment,
                 COALESCE(evaluation_version, 'legacy'),
                 COALESCE(scoring_version, 'legacy'),
                 COALESCE(experiment_id, 'untracked')
        ORDER BY domain, avg_f1 DESC
        """
    ).fetchall()
    if not rows:
        return

    print("\nInterpretation")
    print("--------------")
    for domain in sorted({row[0] for row in rows}):
        domain_rows = [row for row in rows if row[0] == domain]
        best = domain_rows[0]
        model_assignment = _json_loads(best[2], {})
        if model_assignment and len(set(model_assignment.values())) == 1:
            models = next(iter(model_assignment.values()))
        else:
            models = best[2]
        print(
            f"{domain}: best avg_f1 is {best[3]:.4f} for {best[1]} / {models}; "
            f"avg_retrieval_recall={best[4]:.4f}, hallucination_rate={best[5]:.4f}, "
            f"avg_latency_ms={best[6]:.2f}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark results.db without sqlite3 CLI.")
    parser.add_argument(
        "--db",
        type=Path,
        default=RESULTS_DB_PATH,
        help=f"Path to results database. Default: {RESULTS_DB_PATH}",
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=0,
        help="Also print the most recent N individual result rows.",
    )
    parser.add_argument(
        "--failures",
        action="store_true",
        help="Also print grouped zero-F1, zero-retrieval, retry, and hallucination counts.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full metric table including min/max latency, costs, and node input token averages.",
    )
    parser.add_argument(
        "--interpret",
        action="store_true",
        help="Also print a short best-F1 interpretation per domain.",
    )
    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Include repeated question rows in aggregates (default: latest only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"Results database not found: {args.db}")

    with sqlite3.connect(args.db) as conn:
        _ensure_analysis_columns(conn)
        _prepare_analysis_view(conn, include_duplicates=args.include_duplicates)
        if not args.include_duplicates:
            print("Aggregates use the latest row per unique question; duplicates are excluded.\n")
        headers, rows = _fetch(conn, SUMMARY_QUERY if args.full else COMPACT_QUERY)
        _print_table(headers, rows)

        if args.failures:
            print()
            headers, rows = _fetch(conn, FAILURE_QUERY)
            _print_table(headers, rows)

        if args.recent > 0:
            print()
            headers, rows = _fetch(conn, RECENT_QUERY, (args.recent,))
            _print_table(headers, rows)

        if args.interpret:
            _print_recommendations(conn)


if __name__ == "__main__":
    main()
