from __future__ import annotations

import argparse
import json
import sqlite3
import textwrap
from typing import Any

from config.domains import RESULTS_DB_PATH
from evaluation.runner import _load_pairs


def _json_loads(value: object, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _wrap(value: object, width: int = 100) -> str:
    text = str(value or "")
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def _section(title: str, value: object) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(_wrap(value))


def _lookup_pair(domain: str, question_id: str) -> tuple[str, str, list[str]]:
    try:
        pairs = _load_pairs(domain, None)
    except Exception:
        return "", "", []
    for pair in pairs:
        if pair.question_id == question_id:
            return pair.question, pair.answer, pair.evidence
    return "", "", []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one stored benchmark result row.")
    parser.add_argument("id", type=int, help="Result row id from evaluation/results.db.")
    parser.add_argument(
        "--chunks",
        type=int,
        default=5,
        help="Number of retrieved chunks to print. Default: 5.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=900,
        help="Maximum characters per retrieved chunk. Default: 900.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with sqlite3.connect(RESULTS_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM results WHERE id = ?", (args.id,)).fetchone()

    if row is None:
        raise ValueError(f"No result row found with id={args.id}")

    row_dict = dict(row)
    question = row_dict.get("original_query") or ""
    gold_answer = row_dict.get("gold_answer") or ""
    gold_evidence = _json_loads(row_dict.get("gold_evidence"), [])

    if not question or not gold_evidence:
        fallback_question, fallback_answer, fallback_evidence = _lookup_pair(
            str(row_dict["domain"]), str(row_dict["question_id"])
        )
        question = question or fallback_question
        gold_answer = gold_answer or fallback_answer
        gold_evidence = gold_evidence or fallback_evidence

    retrieved_chunks = _json_loads(row_dict.get("retrieved_chunks"), [])
    retrieved_metadata = _json_loads(row_dict.get("retrieved_metadata"), [])
    verification_result = _json_loads(row_dict.get("verification_result"), {})

    print(
        " | ".join(
            [
                f"id={row_dict['id']}",
                f"domain={row_dict['domain']}",
                f"config={row_dict['config_name']}",
                f"question_id={row_dict['question_id']}",
                f"evaluation={row_dict.get('evaluation_version') or 'legacy'}",
                f"f1={row_dict['f1_score']}",
                f"f2={row_dict.get('f2_score')}",
                f"rouge_l={row_dict.get('rouge_l_f1')}",
                f"semantic={row_dict.get('semantic_similarity')}",
                f"precision={row_dict.get('answer_precision')}",
                f"recall={row_dict.get('answer_recall')}",
                f"verbosity={row_dict.get('verbosity_ratio')}",
                f"retrieval_recall={row_dict['retrieval_recall']}",
                f"retries={row_dict['retry_count']}",
                f"hallucination={row_dict['hallucination']}",
                f"execution_latency_ms={row_dict.get('execution_latency_ms')}",
                f"observed_latency_ms={row_dict.get('total_latency_ms')}",
                f"quota_wait_ms={row_dict.get('rate_limit_wait_ms')}",
                f"groq_tokens={row_dict.get('groq_total_tokens')}",
            ]
        )
    )

    _section("Models", row_dict.get("model_assignment"))
    _section(
        "Groq Rate-Limit Policy",
        json.dumps(_json_loads(row_dict.get("rate_limit_policy"), {}), indent=2),
    )
    _section("Question", question)
    _section("Rewritten Query", row_dict.get("rewritten_query"))
    gold_answers = _json_loads(row_dict.get("gold_answers"), [gold_answer])
    _section("Reference Answers", json.dumps(gold_answers, indent=2))
    _section(
        "Reference Answer Types",
        json.dumps(_json_loads(row_dict.get("gold_answer_types"), []), indent=2),
    )
    _section("Answer Type", row_dict.get("answer_type") or "unknown")
    _section("Generated Answer", row_dict.get("synthesized_answer"))
    _section("Gold Evidence", json.dumps(gold_evidence, indent=2))
    _section("Verifier Result", json.dumps(verification_result, indent=2))
    _section(
        "Post-run Evaluation",
        json.dumps(
            {
                "exact_match": row_dict.get("em_score"),
                "max_reference_f1": row_dict.get("f1_score"),
                "max_reference_f2": row_dict.get("f2_score"),
                "max_reference_rouge_l_f1": row_dict.get("rouge_l_f1"),
                "semantic_similarity": row_dict.get("semantic_similarity"),
                "semantic_model": row_dict.get("semantic_model"),
                "categorical_reference_accuracy": row_dict.get("type_accuracy"),
                "answer_precision": row_dict.get("answer_precision"),
                "answer_recall": row_dict.get("answer_recall"),
                "reference_contained": row_dict.get("reference_contained"),
                "verbosity_ratio": row_dict.get("verbosity_ratio"),
                "correctness_pass": row_dict.get("correctness_pass"),
                "scoring_version": row_dict.get("scoring_version"),
                "verifier_parse_success": row_dict.get("verifier_parse_success"),
                "failure_category": row_dict.get("failure_category"),
            },
            indent=2,
        ),
    )

    print("\nRetrieved Chunks")
    print("----------------")
    if not retrieved_chunks:
        print("No retrieved chunks stored for this row. Re-run evaluation to populate debug data.")
        return

    for index, chunk in enumerate(retrieved_chunks[: args.chunks]):
        metadata = retrieved_metadata[index] if index < len(retrieved_metadata) else {}
        print(f"\n[{index}] metadata={json.dumps(metadata, ensure_ascii=True)}")
        print(_wrap(str(chunk)[: args.chunk_chars]))


if __name__ == "__main__":
    main()
