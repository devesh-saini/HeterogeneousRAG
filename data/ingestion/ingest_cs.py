from __future__ import annotations

from datasets import load_dataset

from config.domains import get_domain
from data.ingestion.common import embed_corpus


def _extract_document(row: dict) -> str:
    title = row.get("title", "")
    abstract = row.get("abstract", "")
    sections = row.get("full_text", {})
    if isinstance(sections, dict):
        body = " ".join(
            " ".join(section) if isinstance(section, list) else str(section)
            for section in sections.values()
        )
    else:
        body = str(sections)
    return f"{title}\n{abstract}\n{body}"


def ingest() -> int:
    dataset = load_dataset(get_domain("cs").dataset_id)
    split = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    return embed_corpus(get_domain("cs"), (_extract_document(row) for row in split))


if __name__ == "__main__":
    print(f"Embedded {ingest()} chunks")
