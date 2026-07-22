from __future__ import annotations

from pipeline.nodes.common import add_node_metadata, invoke_llm, load_prompt, render_prompt
from pipeline.state import RAGState


def synthesizer_node(state: RAGState) -> RAGState:
    model_id = state["model_assignment"]["synthesizer"]
    prompt = render_prompt(
        load_prompt(state["domain"], "synthesizer"),
        rewritten_query=state["rewritten_query"],
        evidence=state.get("reranked_chunks", []),
    )
    answer, latency_ms = invoke_llm(model_id, prompt)
    metadata = add_node_metadata(
        state,
        "synthesizer",
        model_id=model_id,
        input_text=prompt,
        output_text=answer,
        latency_ms=latency_ms,
    )
    return {"synthesized_answer": answer, "metadata": metadata}
