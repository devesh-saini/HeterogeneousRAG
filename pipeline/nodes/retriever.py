from __future__ import annotations

import time

from typing import TYPE_CHECKING

from config.domains import get_domain
from pipeline.nodes.common import add_node_metadata
from pipeline.state import RAGState

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


_EMBEDDER: "SentenceTransformer | None" = None


def get_embedder() -> "SentenceTransformer":
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def retriever_node(state: RAGState, top_k: int = 5) -> RAGState:
    domain = get_domain(state["domain"])
    query = state["rewritten_query"]
    start = time.perf_counter()
    import chromadb

    client = chromadb.PersistentClient(path=str(domain.vectordb_path))
    collection = client.get_or_create_collection(domain.collection_name)
    embedding = get_embedder().encode([query], normalize_embeddings=True).tolist()[0]
    result = collection.query(
        query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas"]
    )
    chunks = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    latency_ms = int((time.perf_counter() - start) * 1000)
    metadata = add_node_metadata(
        state,
        "retriever",
        model_id=state["model_assignment"].get("retriever"),
        input_text=query,
        output_text="\n\n".join(chunks),
        latency_ms=latency_ms,
        extra={"top_k": top_k, "retrieved_metadata": metadatas},
    )
    return {"retrieved_chunks": chunks, "retrieved_metadata": metadatas, "metadata": metadata}
