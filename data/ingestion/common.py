from __future__ import annotations

import hashlib
from collections.abc import Iterable

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


def embed_corpus(domain: DomainSpec, documents: Iterable[str], batch_size: int = 64) -> int:
    domain.vectordb_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(domain.vectordb_path))
    collection = client.get_or_create_collection(domain.collection_name)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    chunks = [chunk for doc in dedupe_texts(documents) for chunk in chunk_text(doc)]
    for start in tqdm(range(0, len(chunks), batch_size), desc=f"Embedding {domain.name}"):
        batch = chunks[start : start + batch_size]
        embeddings = embedder.encode(batch, normalize_embeddings=True).tolist()
        ids = [stable_id(chunk) for chunk in batch]
        metadatas = [{"domain": domain.name, "chunk_index": start + i} for i in range(len(batch))]
        collection.upsert(ids=ids, documents=batch, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)
