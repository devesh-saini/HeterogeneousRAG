from __future__ import annotations

import argparse

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
    return parser.parse_args()


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass
    args = parse_args()
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
