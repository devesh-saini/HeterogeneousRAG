from __future__ import annotations

import argparse
import os

from config.domains import DOMAINS
from config.models import MODEL_CONFIGS, get_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heterogeneous RAG benchmark experiments.")
    parser.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    parser.add_argument("--config", "--connfig", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--n_questions", type=int, default=None)
    parser.add_argument(
        "--experiment_id",
        default=None,
        help="Shared id for matched homogeneous/heterogeneous runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip questions already stored for this experiment/config (or latest untracked run).",
    )
    parser.add_argument(
        "--no_semantic_scoring",
        action="store_true",
        help="Disable the lightweight embedding-similarity diagnostic.",
    )
    parser.add_argument(
        "--groq_tpm_limit",
        type=int,
        default=int(os.getenv("GROQ_TPM_LIMIT", "8000")),
        help="Shared Groq token-per-minute ceiling; 0 disables client-side pacing.",
    )
    parser.add_argument(
        "--groq_rate_limit_utilization",
        type=float,
        default=float(os.getenv("GROQ_RATE_LIMIT_UTILIZATION", "0.90")),
        help="Fraction of the Groq TPM ceiling available to the rolling limiter.",
    )
    parser.add_argument(
        "--groq_max_429_retries",
        type=int,
        default=int(os.getenv("GROQ_MAX_429_RETRIES", "4")),
        help="Maximum rate-limit retries per Groq call.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass
    args = parse_args()
    from pipeline.rate_limit import configure_groq_rate_limit

    settings = configure_groq_rate_limit(
        tokens_per_minute=args.groq_tpm_limit,
        utilization=args.groq_rate_limit_utilization,
        max_429_retries=args.groq_max_429_retries,
    )
    if settings.enabled:
        print(
            "Groq pacing: "
            f"profile={settings.profile}, "
            f"rolling_budget={settings.working_token_budget} tokens/"
            f"{settings.window_seconds:.0f}s"
        )
    else:
        print("Groq pacing: disabled")
    model_assignment = get_model_config(args.config)
    from evaluation.runner import run_evaluation

    run_evaluation(
        domain=args.domain,
        config_name=args.config,
        model_assignment=model_assignment,
        n_questions=args.n_questions,
        experiment_id=args.experiment_id,
        resume=args.resume,
        semantic_scoring=not args.no_semantic_scoring,
    )


if __name__ == "__main__":
    main()
