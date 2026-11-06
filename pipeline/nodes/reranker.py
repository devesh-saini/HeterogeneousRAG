from __future__ import annotations

import json

from config.models import RERANKER_PROMPT_CHARACTER_BUDGET, ROLE_MAX_OUTPUT_TOKENS
from pipeline.nodes.common import add_node_metadata, invoke_llm, load_prompt, render_prompt
from pipeline.state import RAGState


def _fallback_rank(chunks: list[str]) -> list[str]:
    return chunks[:3]


def _bounded_prompt_chunks(chunks: list[str]) -> tuple[list[str], int]:
    if not chunks:
        return [], 0
    per_chunk_budget = max(200, RERANKER_PROMPT_CHARACTER_BUDGET // len(chunks))
    prompt_chunks: list[str] = []
    truncated = 0
    marker = "\n...[middle truncated for token budget]...\n"
    for chunk in chunks:
        if len(chunk) <= per_chunk_budget:
            prompt_chunks.append(chunk)
            continue
        truncated += 1
        available = max(1, per_chunk_budget - len(marker))
        head_size = available * 2 // 3
        tail_size = available - head_size
        prompt_chunks.append(f"{chunk[:head_size]}{marker}{chunk[-tail_size:]}")
    return prompt_chunks, truncated


def reranker_node(state: RAGState) -> RAGState:
    model_id = state["model_assignment"]["reranker"]
    chunks = state.get("retrieved_chunks", [])
    prompt_chunks, truncated_chunks = _bounded_prompt_chunks(chunks)
    prompt = render_prompt(
        load_prompt(state["domain"], "reranker"),
        rewritten_query=state["rewritten_query"],
        chunks=[{"index": i, "text": chunk} for i, chunk in enumerate(prompt_chunks)],
    )
    output, latency_ms, invocation_metadata = invoke_llm(
        model_id,
        prompt,
        max_tokens=ROLE_MAX_OUTPUT_TOKENS["reranker"],
    )
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
        extra={
            **invocation_metadata,
            "selected": len(ranked_chunks),
            "prompt_truncated_chunks": truncated_chunks,
            "prompt_character_budget": RERANKER_PROMPT_CHARACTER_BUDGET,
        },
    )
    return {"reranked_chunks": ranked_chunks, "metadata": metadata}
