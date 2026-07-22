from __future__ import annotations

from datasets import load_dataset

from config.domains import get_domain
from data.ingestion.common import embed_corpus


def _extract_context(row: dict) -> str:
    return str(row.get("context") or row.get("contract") or row.get("text") or "")


def ingest() -> int:
    dataset = load_dataset(get_domain("law").dataset_id)
    split = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    return embed_corpus(get_domain("law"), (_extract_context(row) for row in split))


if __name__ == "__main__":
    print(f"Embedded {ingest()} chunks")
