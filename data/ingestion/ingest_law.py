from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.domains import get_domain
from data.ingestion.common import IngestDocument, embed_corpus
from data.ingestion.sources import load_cuad_documents


def ingest() -> int:
    return embed_corpus(
        get_domain("law"),
        (
            IngestDocument(
                str(paragraph.get("context", "")),
                {
                    "source_split": split,
                    "source_id": str(document.get("title", "")),
                },
            )
            for split in ("train", "test")
            for document in load_cuad_documents(split).get("data", [])
            if isinstance(document, dict)
            for paragraph in document.get("paragraphs", [])
            if isinstance(paragraph, dict)
        ),
    )


if __name__ == "__main__":
    print(f"Embedded {ingest()} chunks")
