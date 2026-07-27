from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.domains import get_domain
from data.ingestion.common import embed_corpus
from data.ingestion.sources import load_qasper_documents


def _extract_document(row: dict) -> str:
    title = row.get("title", "")
    abstract = row.get("abstract", "")
    sections = row.get("full_text", {})
    if isinstance(sections, list):
        body = " ".join(
            " ".join(section.get("paragraphs", []))
            if isinstance(section, dict)
            else str(section)
            for section in sections
        )
    elif isinstance(sections, dict) and "paragraphs" in sections:
        paragraphs = sections.get("paragraphs", [])
        if paragraphs and all(isinstance(item, list) for item in paragraphs):
            body = " ".join(" ".join(item) for item in paragraphs)
        elif isinstance(paragraphs, list):
            body = " ".join(str(item) for item in paragraphs)
        else:
            body = str(paragraphs)
    elif isinstance(sections, dict):
        body = " ".join(
            " ".join(section) if isinstance(section, list) else str(section)
            for section in sections.values()
        )
    else:
        body = str(sections)
    return f"{title}\n{abstract}\n{body}"


def ingest() -> int:
    return embed_corpus(
        get_domain("computerScience"), (_extract_document(row) for row in load_qasper_documents())
    )


if __name__ == "__main__":
    print(f"Embedded {ingest()} chunks")
