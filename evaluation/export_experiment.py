from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from config.domains import RESULTS_DB_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export immutable protocol manifests for a named experiment."
    )
    parser.add_argument("experiment_id")
    parser.add_argument("--db", type=Path, default=RESULTS_DB_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='experiment_manifests'"
        ).fetchone()
        if not table_exists:
            raise ValueError(
                "This database has no experiment manifests; run a named experiment first."
            )
        rows = connection.execute(
            "SELECT * FROM experiment_manifests WHERE experiment_id = ? "
            "ORDER BY domain, config_name",
            (args.experiment_id,),
        ).fetchall()
    if not rows:
        raise ValueError(f"No manifest found for experiment '{args.experiment_id}'.")

    json_columns = {
        "model_assignment",
        "question_ids",
        "corpus_manifest",
        "runtime_environment",
        "rate_limit_policy",
    }
    manifests: list[dict[str, object]] = []
    for row in rows:
        manifest = dict(row)
        for column in json_columns:
            manifest[column] = json.loads(str(manifest[column]))
        manifests.append(manifest)
    payload = {
        "schema": "heterogeneous-rag-experiment-manifest-v1",
        "experiment_id": args.experiment_id,
        "manifests": manifests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported {len(manifests)} manifest(s) to {args.output}")


if __name__ == "__main__":
    main()
