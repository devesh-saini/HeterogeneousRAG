from __future__ import annotations


EVALUATION_VERSION = "2.0-reference-aware"

# Additive only: historical result rows remain untouched and are labelled legacy.
RESULT_COLUMN_MIGRATIONS: dict[str, str] = {
    "original_query": "TEXT",
    "rewritten_query": "TEXT",
    "gold_evidence": "TEXT",
    "retrieved_chunks": "TEXT",
    "retrieved_metadata": "TEXT",
    "verification_result": "TEXT",
    "gold_answers": "TEXT",
    "answer_type": "TEXT",
    "answer_precision": "REAL",
    "answer_recall": "REAL",
    "reference_contained": "INTEGER",
    "verbosity_ratio": "REAL",
    "matched_reference_index": "INTEGER",
    "reference_count": "INTEGER",
    "groundedness_pass": "INTEGER",
    "relevance_pass": "INTEGER",
    "completeness_pass": "INTEGER",
    "verifier_parse_success": "INTEGER",
    "correctness_pass": "INTEGER",
    "failure_category": "TEXT",
    "evaluation_version": "TEXT",
}
