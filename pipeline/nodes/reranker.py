from __future__ import annotations

import json

from pipeline.nodes.common import add_node_metadata, invoke_llm, load_prompt, render_prompt
from pipeline.state import RAGState


def _fallback_rank(chunks: list[str]) -> list[str]:
    return chunks[:3]


def reranker_node(state: RAGState) -> RAGState:
    model_id = state["model_assignment"]["reranker"]
    chunks = state.get("retrieved_chunks", [])
    prompt = render_prompt(
        load_prompt(state["domain"], "reranker"),
        rewritten_query=state["rewritten_query"],
        chunks=[{"index": i, "text": chunk} for i, chunk in enumerate(chunks)],
    )
    output, latency_ms = invoke_llm(model_id, prompt)
    try:
        ranked_indexes = json.loads(output)
        ranked_chunks = [chunks[int(i)] for i in ranked_indexes if 0 <= int(i) < len(chunks)][:3]
    except Exception:
        ranked_chunks = _fallback_rank(chunks)
    metadata = add_node_metadata(
        state,
        "reranker",
        model_id=model_id,
        input_text=prompt,
        output_text=output,
        latency_ms=latency_ms,
        extra={"selected": len(ranked_chunks)},
    )
    return {"reranked_chunks": ranked_chunks, "metadata": metadata}
