from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.domains import get_domain
from data.ingestion.common import IngestDocument, embed_corpus
from data.ingestion.sources import load_cuad_contexts


def ingest() -> int:
    return embed_corpus(
        get_domain("law"),
        (
            IngestDocument(context, {"source_split": split})
            for split in ("train", "test")
            for context in load_cuad_contexts(split)
        ),
    )


if __name__ == "__main__":
    print(f"Embedded {ingest()} chunks")
