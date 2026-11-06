from __future__ import annotations

from config.models import ROLE_MAX_OUTPUT_TOKENS
from pipeline.nodes.common import add_node_metadata, invoke_llm, load_prompt, render_prompt
from pipeline.state import RAGState


def rewriter_node(state: RAGState) -> RAGState:
    model_id = state["model_assignment"]["rewriter"]
    previous_verification = state.get("verification_result", {})
    retry_count = int(state.get("retry_count", 0))
    if previous_verification:
        retry_count += 1

    prompt = render_prompt(
        load_prompt(state["domain"], "rewriter"),
        original_query=state["original_query"],
        retry_count=retry_count,
        previous_failure_reasoning=previous_verification.get("reasoning", ""),
    )
    rewritten_query, latency_ms, invocation_metadata = invoke_llm(
        model_id,
        prompt,
        max_tokens=ROLE_MAX_OUTPUT_TOKENS["rewriter"],
    )
    metadata = add_node_metadata(
        state,
        "rewriter",
        model_id=model_id,
        input_text=prompt,
        output_text=rewritten_query,
        latency_ms=latency_ms,
        extra=invocation_metadata,
    )
    return {"rewritten_query": rewritten_query, "retry_count": retry_count, "metadata": metadata}
