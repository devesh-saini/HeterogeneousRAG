from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from datasets import load_dataset

from config.domains import get_domain
from data.ingestion.common import embed_corpus


def _extract_document(row: dict) -> str:
    return str(row.get("passage") or row.get("context") or row.get("document") or row)


def ingest() -> int:
    dataset = load_dataset(get_domain("medical").dataset_id, "text-corpus")
    split = dataset["passages"] if "passages" in dataset else next(iter(dataset.values()))
    return embed_corpus(get_domain("medical"), (_extract_document(row) for row in split))


if __name__ == "__main__":
    print(f"Embedded {ingest()} chunks")
