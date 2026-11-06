from __future__ import annotations

import argparse
import itertools
import math
import random
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path

from config.domains import RESULTS_DB_PATH
from evaluation.schema import SCORING_VERSION


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str
    higher_is_better: bool = True


METRICS = (
    MetricSpec("f1_score", "Official answer F1"),
    MetricSpec("f2_score", "Answer F2 (coverage)"),
    MetricSpec("rouge_l_f1", "ROUGE-L F1"),
    MetricSpec("semantic_similarity", "Semantic similarity"),
    MetricSpec("type_accuracy", "Categorical-reference accuracy"),
    MetricSpec("correctness_pass", "Correctness pass rate"),
    MetricSpec("retrieval_recall", "Retrieval recall"),
    MetricSpec("groundedness_pass", "Groundedness rate"),
    MetricSpec("relevance_pass", "Relevance rate"),
    MetricSpec("completeness_pass", "Completeness rate"),
    MetricSpec("retry_count", "Retries", False),
    MetricSpec("execution_latency_ms", "Execution latency excl. quota wait (ms)", False),
    MetricSpec("total_latency_ms", "Observed latency incl. quota wait (ms)", False),
)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_ci(
    differences: list[float], repetitions: int, seed: int
) -> tuple[float, float]:
    if not differences:
        return math.nan, math.nan
    generator = random.Random(seed)
    size = len(differences)
    means = sorted(
        sum(differences[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(repetitions)
    )
    return _percentile(means, 0.025), _percentile(means, 0.975)


def paired_randomization_pvalue(
    differences: list[float], repetitions: int, seed: int
) -> float:
    nonzero = [difference for difference in differences if abs(difference) > 1e-12]
    if not nonzero:
        return 1.0
    observed = abs(sum(nonzero))
    if len(nonzero) <= 20:
        extreme = 0
        total = 1 << len(nonzero)
        for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
            permuted = abs(sum(sign * value for sign, value in zip(signs, nonzero)))
            extreme += permuted >= observed - 1e-12
        return extreme / total

    generator = random.Random(seed)
    extreme = 0
    for _ in range(repetitions):
        permuted = abs(
            sum(value if generator.random() < 0.5 else -value for value in nonzero)
        )
        extreme += permuted >= observed - 1e-12
    return (extreme + 1) / (repetitions + 1)


def _effect_size(differences: list[float]) -> float:
    if len(differences) < 2:
        return math.nan
    standard_deviation = statistics.stdev(differences)
    if standard_deviation == 0:
        return math.inf if statistics.mean(differences) else 0.0
    return statistics.mean(differences) / standard_deviation


def _approximate_pairs_for_power(effect_size: float) -> str:
    """Normal approximation for 80% power at two-sided alpha=0.05."""
    if math.isnan(effect_size) or effect_size == 0:
        return "n/a"
    if math.isinf(effect_size):
        return "2"
    required = math.ceil(((1.96 + 0.84) / abs(effect_size)) ** 2)
    return ">9999" if required > 9999 else str(max(2, required))


def _format(value: float, metric: MetricSpec) -> str:
    if math.isnan(value):
        return "n/a"
    if metric.column.endswith("latency_ms"):
        return f"{value:.0f}"
    return f"{value:.4f}"


def _load_latest(
    connection: sqlite3.Connection,
    *,
    domain: str,
    scoring_version: str,
    experiment_id: str | None,
    configs: tuple[str, str],
    rate_limit_profile: str | None,
) -> dict[str, dict[str, sqlite3.Row]]:
    available_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(results)").fetchall()
    }
    where = ["domain = ?", "scoring_version = ?", "config_name IN (?, ?)"]
    params: list[object] = [domain, scoring_version, *configs]
    if experiment_id:
        where.append("experiment_id = ?")
        params.append(experiment_id)
    if rate_limit_profile:
        if "rate_limit_profile" in available_columns:
            where.append("COALESCE(rate_limit_profile, 'legacy-unpaced') = ?")
            params.append(rate_limit_profile)
        elif rate_limit_profile != "legacy-unpaced":
            return {config: {} for config in configs}
    compatibility_columns = [
        column
        for column in ("execution_latency_ms", "rate_limit_profile")
        if column not in available_columns
    ]
    compatibility_select = "".join(
        f", NULL AS {column}" for column in compatibility_columns
    )
    rows = connection.execute(
        f"SELECT *{compatibility_select} FROM results "
        f"WHERE {' AND '.join(where)} ORDER BY id",
        params,
    ).fetchall()
    latest: dict[str, dict[str, sqlite3.Row]] = {config: {} for config in configs}
    for row in rows:
        latest[str(row["config_name"])][str(row["question_id"])] = row
    return latest


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired, duplicate-safe comparison of two benchmark configurations."
    )
    parser.add_argument("--db", type=Path, default=RESULTS_DB_PATH)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--a", default="homogeneous", help="Baseline configuration.")
    parser.add_argument("--b", default="heterogeneous", help="Comparison configuration.")
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--rate-limit-profile",
        help="Restrict both configurations to one recorded pacing profile.",
    )
    parser.add_argument("--scoring-version", default=SCORING_VERSION)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        latest = _load_latest(
            connection,
            domain=args.domain,
            scoring_version=args.scoring_version,
            experiment_id=args.experiment_id,
            configs=(args.a, args.b),
            rate_limit_profile=args.rate_limit_profile,
        )

    shared_ids = sorted(set(latest[args.a]).intersection(latest[args.b]))
    if not shared_ids:
        raise ValueError("No paired question IDs found for the requested comparison.")

    print(
        f"Paired comparison: {args.a} (A) vs {args.b} (B) | "
        f"domain={args.domain} | n={len(shared_ids)} | scoring={args.scoring_version}"
    )
    if args.experiment_id:
        print(f"Experiment: {args.experiment_id}")
    else:
        print("Selection: latest row per config/question (duplicates excluded)")
    profiles = sorted(
        {
            str(latest[config][question_id]["rate_limit_profile"] or "legacy-unpaced")
            for config in (args.a, args.b)
            for question_id in shared_ids
        }
    )
    print(f"Rate-limit profile(s): {', '.join(profiles)}")
    if len(profiles) > 1:
        print(
            "WARNING: multiple pacing profiles are present. Use a shared experiment ID "
            "or --rate-limit-profile for a protocol-consistent latency comparison."
        )

    output: list[list[str]] = []
    for metric_index, metric in enumerate(METRICS):
        pairs = [
            (latest[args.a][question_id][metric.column], latest[args.b][question_id][metric.column])
            for question_id in shared_ids
        ]
        numeric_pairs = [
            (float(left), float(right))
            for left, right in pairs
            if left is not None and right is not None
        ]
        if not numeric_pairs:
            continue
        left_values = [pair[0] for pair in numeric_pairs]
        right_values = [pair[1] for pair in numeric_pairs]
        raw_differences = [right - left for left, right in numeric_pairs]
        preferred_differences = [
            difference if metric.higher_is_better else -difference
            for difference in raw_differences
        ]
        lower, upper = bootstrap_ci(
            preferred_differences, args.bootstrap, args.seed + metric_index
        )
        pvalue = paired_randomization_pvalue(
            preferred_differences, args.permutations, args.seed + metric_index
        )
        wins = sum(difference > 1e-12 for difference in preferred_differences)
        ties = sum(abs(difference) <= 1e-12 for difference in preferred_differences)
        losses = sum(difference < -1e-12 for difference in preferred_differences)
        baseline = statistics.mean(left_values)
        comparison = statistics.mean(right_values)
        improvement = statistics.mean(preferred_differences)
        relative = improvement / abs(baseline) * 100 if baseline else math.nan
        clear = lower > 0 and pvalue < 0.05
        effect = _effect_size(preferred_differences)
        output.append(
            [
                metric.label,
                str(len(numeric_pairs)),
                _format(baseline, metric),
                _format(comparison, metric),
                _format(improvement, metric),
                "n/a" if math.isnan(relative) else f"{relative:+.1f}%",
                f"[{_format(lower, metric)}, {_format(upper, metric)}]",
                "inf" if math.isinf(effect) else f"{effect:.2f}",
                _approximate_pairs_for_power(effect),
                f"{wins}/{ties}/{losses}",
                f"{pvalue:.4f}",
                "yes" if clear else "no",
            ]
        )

    _print_table(
        [
            "Metric",
            "n",
            "A mean",
            "B mean",
            "B improvement",
            "Relative",
            "95% bootstrap CI",
            "paired d",
            "n@80%*",
            "W/T/L",
            "p",
            "clear?",
        ],
        output,
    )

    print("\nOfficial answer F1 by answer type")
    type_rows: list[list[str]] = []
    answer_types = sorted(
        {
            str(
                latest[config][question_id]["matched_answer_type"]
                or latest[config][question_id]["answer_type"]
                or "unknown"
            )
            for config in (args.a, args.b)
            for question_id in shared_ids
        }
    )
    for answer_type in answer_types:
        ids = [
            question_id
            for question_id in shared_ids
            if str(
                latest[args.a][question_id]["matched_answer_type"]
                or latest[args.a][question_id]["answer_type"]
                or "unknown"
            ) == answer_type
            and str(
                latest[args.b][question_id]["matched_answer_type"]
                or latest[args.b][question_id]["answer_type"]
                or "unknown"
            ) == answer_type
        ]
        if not ids:
            continue
        left = statistics.mean(float(latest[args.a][qid]["f1_score"]) for qid in ids)
        right = statistics.mean(float(latest[args.b][qid]["f1_score"]) for qid in ids)
        type_rows.append([answer_type, str(len(ids)), f"{left:.4f}", f"{right:.4f}", f"{right-left:+.4f}"])
    _print_table(["Answer type", "n", "A F1", "B F1", "B-A"], type_rows)

    print(
        "\nInterpretation rule: 'clear=yes' requires the paired bootstrap CI to be "
        "above zero and the two-sided paired randomization p-value to be below 0.05. "
        "B improvement is direction-adjusted, so positive always favors B. "
        "*n@80% is a rough normal-approximation based on the observed paired effect; "
        "small-sample estimates can be unstable. P-values are unadjusted exploratory results."
    )


if __name__ == "__main__":
    main()
