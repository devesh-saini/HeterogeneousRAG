from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from config.domains import DomainSpec

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: object):
        return iterable


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IngestDocument:
    text: str
    metadata: dict[str, Any] | None = None


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def dedupe_texts(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for text in texts:
        normalized = " ".join(str(text).split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _metadata_value(value: Any) -> str | int | float | bool | None:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _iter_chunk_records(
    documents: Iterable[str | IngestDocument],
) -> list[tuple[str, dict[str, str | int | float | bool | None]]]:
    seen: set[str] = set()
    records: list[tuple[str, dict[str, str | int | float | bool | None]]] = []
    for doc_index, document in enumerate(documents):
        if isinstance(document, IngestDocument):
            text = document.text
            base_metadata = document.metadata or {}
        else:
            text = str(document)
            base_metadata = {}

        normalized = " ".join(str(text).split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)

        chunks = chunk_text(normalized)
        for local_chunk_index, chunk in enumerate(chunks):
            metadata = {
                key: _metadata_value(value) for key, value in base_metadata.items()
            }
            metadata.update(
                {
                    "document_index": doc_index,
                    "document_chunk_index": local_chunk_index,
                }
            )
            records.append((chunk, metadata))
    return records


def embed_corpus(
    domain: DomainSpec, documents: Iterable[str | IngestDocument], batch_size: int = 64
) -> int:
    domain.vectordb_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(domain.vectordb_path))
    collection = client.get_or_create_collection(domain.collection_name)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    records = _iter_chunk_records(documents)
    if not records:
        raise ValueError(f"No text chunks were produced for domain '{domain.name}'")
    for start in tqdm(range(0, len(records), batch_size), desc=f"Embedding {domain.name}"):
        batch_records = records[start : start + batch_size]
        batch = [chunk for chunk, _ in batch_records]
        embeddings = embedder.encode(batch, normalize_embeddings=True).tolist()
        ids = [stable_id(chunk) for chunk in batch]
        metadatas = [
            {"domain": domain.name, "chunk_index": start + i, **metadata}
            for i, (_, metadata) in enumerate(batch_records)
        ]
        collection.upsert(ids=ids, documents=batch, embeddings=embeddings, metadatas=metadatas)
    return len(records)
