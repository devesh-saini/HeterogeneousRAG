from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DomainName = Literal["cs", "medical", "law", "finance"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORDB_ROOT = PROJECT_ROOT / "vectordb"
RESULTS_DB_PATH = PROJECT_ROOT / "evaluation" / "results.db"


@dataclass(frozen=True)
class DomainSpec:
    name: DomainName
    dataset_id: str
    vectordb_path: Path
    collection_name: str


DOMAINS: dict[DomainName, DomainSpec] = {
    "cs": DomainSpec("cs", "allenai/qasper", VECTORDB_ROOT / "cs", "cs"),
    "medical": DomainSpec(
        "medical", "rag-datasets/rag-mini-bioasq", VECTORDB_ROOT / "medical", "medical"
    ),
    "law": DomainSpec("law", "theatticusproject/cuad-qa", VECTORDB_ROOT / "law", "law"),
    "finance": DomainSpec(
        "finance", "PatronusAI/financebench", VECTORDB_ROOT / "finance", "finance"
    ),
}


def get_domain(domain: str) -> DomainSpec:
    try:
        return DOMAINS[domain]  # type: ignore[index]
    except KeyError as exc:
        valid = ", ".join(sorted(DOMAINS))
        raise ValueError(f"Unknown domain '{domain}'. Expected one of: {valid}") from exc
