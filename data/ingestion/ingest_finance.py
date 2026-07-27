from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from datasets import load_dataset

from config.domains import get_domain
from data.ingestion.common import IngestDocument, embed_corpus


def _extract_evidence_text(evidence: object) -> str:
    if isinstance(evidence, list):
        values: list[str] = []
        for item in evidence:
            if isinstance(item, dict):
                values.append(str(item.get("evidence_text") or item.get("text") or item))
            else:
                values.append(str(item))
        return "\n\n".join(value for value in values if value.strip())
    if isinstance(evidence, dict):
        return str(evidence.get("evidence_text") or evidence.get("text") or evidence)
    return str(evidence or "")


def _extract_context(row: dict) -> str:
    evidence_text = _extract_evidence_text(row.get("evidence"))
    doc_text = str(row.get("context") or row.get("doc_text") or "")
    return "\n\n".join(part for part in [doc_text, evidence_text] if part.strip())


def ingest() -> int:
    dataset = load_dataset(get_domain("finance").dataset_id)
    split = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
    return embed_corpus(
        get_domain("finance"),
        (
            IngestDocument(
                _extract_context(row),
                {
                    "source_id": str(row.get("financebench_id", "")),
                    "doc_name": str(row.get("doc_name", "")),
                },
            )
            for row in split
        ),
    )


if __name__ == "__main__":
    print(f"Embedded {ingest()} chunks")
