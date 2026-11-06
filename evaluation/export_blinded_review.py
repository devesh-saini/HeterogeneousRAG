from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from pathlib import Path

from config.domains import RESULTS_DB_PATH
from evaluation.compare_configs import _load_latest, _manifest_question_ids
from evaluation.schema import SCORING_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export randomized, blinded answer pairs for independent review."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--a", default="homogeneous")
    parser.add_argument("--b", default="heterogeneous")
    parser.add_argument("--db", type=Path, default=RESULTS_DB_PATH)
    parser.add_argument("--scoring-version", default=SCORING_VERSION)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--review-sheet", type=Path, required=True)
    parser.add_argument("--assignment-key", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.review_sheet.resolve() == args.assignment_key.resolve():
        raise ValueError("Review sheet and assignment key must be separate files.")
    configs = (args.a, args.b)
    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        latest = _load_latest(
            connection,
            domain=args.domain,
            scoring_version=args.scoring_version,
            experiment_id=args.experiment_id,
            configs=configs,
            rate_limit_profile=None,
        )
        planned = _manifest_question_ids(
            connection,
            experiment_id=args.experiment_id,
            domain=args.domain,
            configs=configs,
            scoring_version=args.scoring_version,
        )
    if not planned:
        raise ValueError("The experiment manifest contains no planned questions.")
    for config in configs:
        missing = planned - set(latest[config])
        if missing:
            raise ValueError(
                f"Cannot export a complete blinded review: {config} is missing "
                f"{len(missing)} planned answer(s)."
            )

    generator = random.Random(args.seed)
    review_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, str]] = []
    question_ids = sorted(planned)
    generator.shuffle(question_ids)
    orders = [
        list(configs) if index % 2 == 0 else list(reversed(configs))
        for index in range(len(planned))
    ]
    generator.shuffle(orders)
    for index, (question_id, order) in enumerate(
        zip(question_ids, orders), start=1
    ):
        first = latest[order[0]][question_id]
        second = latest[order[1]][question_id]
        item_id = f"R{index:04d}"
        review_rows.append(
            {
                "item_id": item_id,
                "question": first["original_query"],
                "reference_answers_json": first["gold_answers"],
                "gold_evidence_json": first["gold_evidence"],
                "answer_1": first["synthesized_answer"],
                "answer_2": second["synthesized_answer"],
                "correctness_1_1_to_5": "",
                "correctness_2_1_to_5": "",
                "groundedness_1_1_to_5": "",
                "groundedness_2_1_to_5": "",
                "completeness_1_1_to_5": "",
                "completeness_2_1_to_5": "",
                "preference_1_2_tie": "",
                "reviewer_confidence_1_to_5": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "item_id": item_id,
                "question_id": question_id,
                "answer_1_config": order[0],
                "answer_1_result_id": str(first["id"]),
                "answer_2_config": order[1],
                "answer_2_result_id": str(second["id"]),
            }
        )

    args.review_sheet.parent.mkdir(parents=True, exist_ok=True)
    args.assignment_key.parent.mkdir(parents=True, exist_ok=True)
    with args.review_sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    with args.assignment_key.open("w", newline="", encoding="utf-8") as handle:
        handle.write(
            f"# experiment_id={args.experiment_id}, domain={args.domain}, "
            f"seed={args.seed}, scoring_version={args.scoring_version}\n"
        )
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    print(
        f"Exported {len(review_rows)} blinded pairs to {args.review_sheet}. "
        f"Keep {args.assignment_key} hidden from reviewers."
    )


if __name__ == "__main__":
    main()
