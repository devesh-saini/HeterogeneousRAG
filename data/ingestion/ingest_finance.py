from __future__ import annotations

from datasets import load_dataset

from config.domains import get_domain
from data.ingestion.common import embed_corpus


def _extract_context(row: dict) -> str:
    return str(row.get("context") or row.get("evidence") or row.get("doc_text") or "")


def ingest() -> int:
    dataset = load_dataset(get_domain("finance").dataset_id)
    split = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    return embed_corpus(get_domain("finance"), (_extract_context(row) for row in split))


if __name__ == "__main__":
    print(f"Embedded {ingest()} chunks")
