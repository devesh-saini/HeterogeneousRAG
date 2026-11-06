from __future__ import annotations

import argparse
import itertools
import json
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
    MetricSpec("groundedness_pass", "Self-verifier groundedness rate"),
    MetricSpec("relevance_pass", "Self-verifier relevance rate"),
    MetricSpec("completeness_pass", "Self-verifier completeness rate"),
    MetricSpec("retry_count", "Retries", False),
    MetricSpec("total_cost_usd", "Recorded API cost (USD)", False),
    MetricSpec("execution_latency_ms", "Execution latency excl. quota wait (ms)", False),
    MetricSpec("total_latency_ms", "Observed latency incl. quota wait (ms)", False),
)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Return Holm-adjusted p-values while preserving the input order."""
    if not pvalues:
        return []
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [0.0] * len(pvalues)
    running_max = 0.0
    count = len(pvalues)
    for rank, (original_index, pvalue) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - rank) * pvalue))
        adjusted[original_index] = running_max
    return adjusted


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


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
    differences: list[float], repetitions: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    if not differences:
        return math.nan, math.nan
    generator = random.Random(seed)
    size = len(differences)
    means = sorted(
        sum(differences[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(repetitions)
    )
    return _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2)


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
    if metric.column == "total_cost_usd":
        return f"{value:.6f}"
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


def _manifest_question_ids(
    connection: sqlite3.Connection,
    *,
    experiment_id: str,
    domain: str,
    configs: tuple[str, str],
    scoring_version: str,
) -> set[str]:
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='experiment_manifests'"
    ).fetchone()
    if not table_exists:
        raise ValueError(
            "Named confirmatory comparisons require experiment manifests. This appears "
            "to be a legacy experiment; rerun it under the current protocol."
        )
    rows = connection.execute(
        """
        SELECT config_name, question_ids, scoring_version
        FROM experiment_manifests
        WHERE experiment_id = ? AND domain = ? AND config_name IN (?, ?)
        """,
        (experiment_id, domain, *configs),
    ).fetchall()
    manifests = {str(row[0]): row for row in rows}
    missing = [config for config in configs if config not in manifests]
    if missing:
        raise ValueError(
            f"Missing experiment manifest(s) for: {', '.join(missing)}. Complete both "
            "configurations under the same named experiment before comparison."
        )
    planned_samples: list[set[str]] = []
    for config in configs:
        row = manifests[config]
        if str(row[2]) != scoring_version:
            raise ValueError(
                f"Manifest scoring version for '{config}' is {row[2]}, not "
                f"the requested {scoring_version}."
            )
        planned_samples.append(set(map(str, json.loads(str(row[1])))))
    if planned_samples[0] != planned_samples[1]:
        raise ValueError("Configuration manifests do not contain the same planned sample.")
    return planned_samples[0]


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
    parser.add_argument(
        "--primary-metric",
        choices=[metric.column for metric in METRICS],
        default="f1_score",
        help="Pre-specified confirmatory endpoint; all others receive Holm correction.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the complete machine-readable analysis to this path.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow exploratory analysis before every planned pair is complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1 or args.permutations < 1:
        raise ValueError("--bootstrap and --permutations must be positive.")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be strictly between 0 and 1.")
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
        planned_ids = (
            _manifest_question_ids(
                connection,
                experiment_id=args.experiment_id,
                domain=args.domain,
                configs=(args.a, args.b),
                scoring_version=args.scoring_version,
            )
            if args.experiment_id
            else None
        )

    manifest_complete = planned_ids is not None and all(
        set(latest[config]) == planned_ids for config in (args.a, args.b)
    )
    confirmatory_protocol_complete = bool(
        manifest_complete
        and args.primary_metric == "f1_score"
        and math.isclose(args.alpha, 0.05)
    )
    if planned_ids is not None and not args.allow_incomplete:
        incomplete: list[str] = []
        for config in (args.a, args.b):
            missing_ids = planned_ids - set(latest[config])
            extra_ids = set(latest[config]) - planned_ids
            if missing_ids or extra_ids:
                incomplete.append(
                    f"{config}: {len(missing_ids)} missing, {len(extra_ids)} unexpected"
                )
        if incomplete:
            raise ValueError(
                "Named experiment is not complete for its planned sample ("
                + "; ".join(incomplete)
                + "). Resume the runs, or use --allow-incomplete only for exploratory analysis."
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
        print(
            "WARNING: no named experiment manifest; this analysis is exploratory "
            "and is not eligible for a confirmatory claim."
        )
    if args.allow_incomplete and not manifest_complete:
        print(
            "WARNING: planned pairs are incomplete; this progress analysis is exploratory."
        )
    if args.experiment_id and manifest_complete and not confirmatory_protocol_complete:
        print(
            "WARNING: primary metric or alpha differs from the repository's fixed "
            "research protocol; treat this analysis as exploratory."
        )
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

    analyses: list[dict[str, object]] = []
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
            preferred_differences,
            args.bootstrap,
            args.seed + metric_index,
            args.alpha,
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
        effect = _effect_size(preferred_differences)
        analyses.append(
            {
                "column": metric.column,
                "label": metric.label,
                "higher_is_better": metric.higher_is_better,
                "n": len(numeric_pairs),
                "a_mean": baseline,
                "b_mean": comparison,
                "preferred_difference": improvement,
                "relative_improvement_percent": relative,
                "bootstrap_ci": [lower, upper],
                "paired_effect_size_dz": effect,
                "approximate_pairs_for_80_percent_power": _approximate_pairs_for_power(effect),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "p_raw": pvalue,
            }
        )

    secondary = [
        analysis for analysis in analyses if analysis["column"] != args.primary_metric
    ]
    if not any(item["column"] == args.primary_metric for item in analyses):
        raise ValueError(
            f"Primary metric '{args.primary_metric}' has no complete paired observations."
        )
    adjusted_secondary = holm_adjust([float(item["p_raw"]) for item in secondary])
    adjusted_by_column = {
        str(item["column"]): adjusted
        for item, adjusted in zip(secondary, adjusted_secondary)
    }
    output: list[list[str]] = []
    for analysis in analyses:
        metric = next(item for item in METRICS if item.column == analysis["column"])
        is_primary = metric.column == args.primary_metric
        adjusted = (
            float(analysis["p_raw"])
            if is_primary
            else adjusted_by_column[metric.column]
        )
        analysis["p_holm"] = adjusted if not is_primary else None
        analysis["endpoint_role"] = "primary" if is_primary else "secondary"
        lower, upper = analysis["bootstrap_ci"]  # type: ignore[misc]
        analysis["evidence_at_alpha"] = bool(lower > 0 and adjusted < args.alpha)
        output.append(
            [
                str(analysis["endpoint_role"]),
                str(analysis["label"]),
                str(analysis["n"]),
                _format(float(analysis["a_mean"]), metric),
                _format(float(analysis["b_mean"]), metric),
                _format(float(analysis["preferred_difference"]), metric),
                (
                    "n/a"
                    if math.isnan(float(analysis["relative_improvement_percent"]))
                    else f"{float(analysis['relative_improvement_percent']):+.1f}%"
                ),
                f"[{_format(lower, metric)}, {_format(upper, metric)}]",
                (
                    "inf"
                    if math.isinf(float(analysis["paired_effect_size_dz"]))
                    else f"{float(analysis['paired_effect_size_dz']):.2f}"
                ),
                str(analysis["approximate_pairs_for_80_percent_power"]),
                f"{analysis['wins']}/{analysis['ties']}/{analysis['losses']}",
                f"{float(analysis['p_raw']):.4f}",
                "--" if is_primary else f"{adjusted:.4f}",
                "yes" if analysis["evidence_at_alpha"] else "no",
            ]
        )

    _print_table(
        [
            "Role",
            "Metric",
            "n",
            "A mean",
            "B mean",
            "B improvement",
            "Relative",
            f"{(1 - args.alpha) * 100:g}% bootstrap CI",
            "paired d",
            "n@80%*",
            "W/T/L",
            "p raw",
            "p Holm",
            "evidence?",
        ],
        output,
    )
    cost_analysis = next(
        (item for item in analyses if item["column"] == "total_cost_usd"), None
    )
    if cost_analysis and not (
        float(cost_analysis["a_mean"]) or float(cost_analysis["b_mean"])
    ):
        print(
            "\nWARNING: recorded cost is zero for both configurations. Do not make a "
            "cost claim unless provider prices and local compute accounting are configured."
        )

    print("\nOfficial answer F1 by answer type")
    type_rows: list[list[str]] = []
    answer_types = sorted(
        {
            str(latest[config][question_id]["answer_type"] or "unknown")
            for config in (args.a, args.b)
            for question_id in shared_ids
        }
    )
    for answer_type in answer_types:
        ids = [
            question_id
            for question_id in shared_ids
            if str(latest[args.a][question_id]["answer_type"] or "unknown")
            == answer_type
            and str(latest[args.b][question_id]["answer_type"] or "unknown")
            == answer_type
        ]
        if not ids:
            continue
        left = statistics.mean(float(latest[args.a][qid]["f1_score"]) for qid in ids)
        right = statistics.mean(float(latest[args.b][qid]["f1_score"]) for qid in ids)
        type_rows.append([answer_type, str(len(ids)), f"{left:.4f}", f"{right:.4f}", f"{right-left:+.4f}"])
    _print_table(["Answer type", "n", "A F1", "B F1", "B-A"], type_rows)

    print(
        f"\nInterpretation rule (alpha={args.alpha:g}): the selected primary "
        "endpoint uses its raw two-sided paired-randomization p-value; secondary "
        "endpoints use Holm-adjusted p-values across the displayed secondary family. "
        "'evidence=yes' also requires the paired bootstrap CI to be above zero. "
        "Positive B improvement always favors B. *n@80% is a rough, post-hoc normal "
        "approximation and must not be presented as an a priori power calculation."
    )

    if args.output_json:
        payload = {
            "comparison": {
                "a": args.a,
                "b": args.b,
                "domain": args.domain,
                "experiment_id": args.experiment_id,
                "scoring_version": args.scoring_version,
                "rate_limit_profile": args.rate_limit_profile,
                "paired_question_count": len(shared_ids),
                "paired_question_ids": shared_ids,
                "primary_metric": args.primary_metric,
                "alpha": args.alpha,
                "confidence_level": 1 - args.alpha,
                "bootstrap_repetitions": args.bootstrap,
                "randomization_repetitions": args.permutations,
                "analysis_seed": args.seed,
                "confirmatory_protocol_complete": confirmatory_protocol_complete,
            },
            "metrics": analyses,
            "answer_type_f1": [
                {
                    "answer_type": row[0],
                    "n": int(row[1]),
                    "a_f1": float(row[2]),
                    "b_f1": float(row[3]),
                    "difference": float(row[4]),
                }
                for row in type_rows
            ],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"Machine-readable analysis: {args.output_json}")


if __name__ == "__main__":
    main()
