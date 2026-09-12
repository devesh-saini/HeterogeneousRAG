from __future__ import annotations


EVALUATION_VERSION = "2.0-reference-aware"
SCORING_VERSION = "3.0-official-plus"
SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Additive only: historical result rows remain untouched and are labelled legacy.
RESULT_COLUMN_MIGRATIONS: dict[str, str] = {
    "original_query": "TEXT",
    "rewritten_query": "TEXT",
    "gold_evidence": "TEXT",
    "retrieved_chunks": "TEXT",
    "retrieved_metadata": "TEXT",
    "verification_result": "TEXT",
    "gold_answers": "TEXT",
    "gold_answer_types": "TEXT",
    "answer_type": "TEXT",
    "answer_precision": "REAL",
    "answer_recall": "REAL",
    "f2_score": "REAL",
    "rouge_l_f1": "REAL",
    "semantic_similarity": "REAL",
    "semantic_model": "TEXT",
    "type_accuracy": "REAL",
    "reference_contained": "INTEGER",
    "verbosity_ratio": "REAL",
    "matched_reference_index": "INTEGER",
    "matched_answer_type": "TEXT",
    "reference_count": "INTEGER",
    "groundedness_pass": "INTEGER",
    "relevance_pass": "INTEGER",
    "completeness_pass": "INTEGER",
    "verifier_parse_success": "INTEGER",
    "correctness_pass": "INTEGER",
    "failure_category": "TEXT",
    "execution_latency_ms": "INTEGER",
    "rate_limit_wait_ms": "INTEGER",
    "groq_input_tokens": "INTEGER",
    "groq_output_tokens": "INTEGER",
    "groq_total_tokens": "INTEGER",
    "groq_requests": "INTEGER",
    "groq_429_retries": "INTEGER",
    "rate_limit_profile": "TEXT",
    "rate_limit_policy": "TEXT",
    "evaluation_version": "TEXT",
    "scoring_version": "TEXT",
    "experiment_id": "TEXT",
}
