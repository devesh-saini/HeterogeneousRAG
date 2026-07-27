from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from config.domains import RESULTS_DB_PATH


SUMMARY_QUERY = """
SELECT
    domain,
    config_name,
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
    ROUND(AVG(retrieval_recall), 4) AS avg_retrieval_recall,
    ROUND(AVG(retry_count), 2) AS avg_retries,
    ROUND(AVG(hallucination), 4) AS hallucination_rate,
    ROUND(AVG(total_cost_usd), 6) AS avg_cost_usd,
    ROUND(AVG(total_latency_ms), 2) AS avg_latency_ms
FROM results
GROUP BY domain, config_name, model_assignment
ORDER BY domain, config_name, models
"""

RECENT_QUERY = """
SELECT
    id,
    domain,
    config_name,
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
    ROUND(retrieval_recall, 4) AS retrieval_recall,
    retry_count,
    hallucination,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.db.exists():
        raise FileNotFoundError(f"Results database not found: {args.db}")

    with sqlite3.connect(args.db) as conn:
        headers, rows = _fetch(conn, SUMMARY_QUERY)
        _print_table(headers, rows)

        if args.recent > 0:
            print()
            headers, rows = _fetch(conn, RECENT_QUERY, (args.recent,))
            _print_table(headers, rows)


if __name__ == "__main__":
    main()
