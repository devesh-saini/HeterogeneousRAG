from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from config.domains import DomainSpec


EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS = 900
CHUNK_OVERLAP_WORDS = 120

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


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
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
    seen: set[tuple[str, str]] = set()
    records: list[tuple[str, dict[str, str | int | float | bool | None]]] = []
    for doc_index, document in enumerate(documents):
        if isinstance(document, IngestDocument):
            text = document.text
            base_metadata = document.metadata or {}
        else:
            text = str(document)
            base_metadata = {}

        normalized = " ".join(str(text).split())
        source_id = str(base_metadata.get("source_id") or "")
        dedupe_key = (source_id, normalized)
        if not normalized or dedupe_key in seen:
            continue
        seen.add(dedupe_key)

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
    records = _iter_chunk_records(documents)
    if not records:
        raise ValueError(f"No text chunks were produced for domain '{domain.name}'")
    # Build in a temporary collection so a failed embedding run does not destroy
    # the last usable corpus. The final swap still guarantees that stale chunks
    # cannot survive into a publication run.
    building_name = f"{domain.collection_name}__building"
    previous_name = f"{domain.collection_name}__previous"
    existing_collections = {
        getattr(collection, "name", str(collection))
        for collection in client.list_collections()
    }
    for stale_name in (building_name, previous_name):
        if stale_name in existing_collections:
            client.delete_collection(stale_name)
    collection = client.create_collection(building_name)
    embedder = SentenceTransformer(EMBEDDING_MODEL_ID)
    for start in tqdm(range(0, len(records), batch_size), desc=f"Embedding {domain.name}"):
        batch_records = records[start : start + batch_size]
        batch = [chunk for chunk, _ in batch_records]
        embeddings = embedder.encode(batch, normalize_embeddings=True).tolist()
        ids = [
            stable_id(json.dumps([metadata.get("source_id"), chunk]))
            for chunk, metadata in batch_records
        ]
        metadatas = [
            {"domain": domain.name, "chunk_index": start + i, **metadata}
            for i, (_, metadata) in enumerate(batch_records)
        ]
        collection.upsert(ids=ids, documents=batch, embeddings=embeddings, metadatas=metadatas)
    if domain.collection_name in existing_collections:
        client.get_collection(domain.collection_name).modify(name=previous_name)
    collection.modify(name=domain.collection_name)
    if domain.collection_name in existing_collections:
        client.delete_collection(previous_name)
    corpus_payload = [
        {"text": chunk, "metadata": metadata} for chunk, metadata in records
    ]
    corpus_sha256 = hashlib.sha256(
        json.dumps(
            corpus_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema": "heterogeneous-rag-corpus-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain": domain.name,
        "dataset_id": domain.dataset_id,
        "collection_name": domain.collection_name,
        "chunk_count": len(records),
        "chunk_size_words": CHUNK_SIZE_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "embedding_model": EMBEDDING_MODEL_ID,
        "corpus_sha256": corpus_sha256,
        "python": platform.python_version(),
    }
    temporary_manifest = domain.vectordb_path / "ingestion_manifest.json.tmp"
    final_manifest = domain.vectordb_path / "ingestion_manifest.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(final_manifest)
    return len(records)
