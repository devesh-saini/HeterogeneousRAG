from __future__ import annotations

import json

from pipeline.nodes.common import add_node_metadata, invoke_llm, load_prompt, render_prompt
from pipeline.state import RAGState


def _parse_verification(output: str) -> dict[str, object]:
    try:
        parsed = json.loads(output)
        return {"verdict": bool(parsed["verdict"]), "reasoning": str(parsed.get("reasoning", ""))}
    except Exception:
        return {"verdict": False, "reasoning": f"Verifier returned non-JSON output: {output[:500]}"}


def verifier_node(state: RAGState) -> RAGState:
    model_id = state["model_assignment"]["verifier"]
    prompt = render_prompt(
        load_prompt(state["domain"], "verifier"),
        answer=state.get("synthesized_answer", ""),
        evidence=state.get("reranked_chunks", []),
    )
    output, latency_ms = invoke_llm(model_id, prompt)
    verification_result = _parse_verification(output)
    metadata = add_node_metadata(
        state,
        "verifier",
        model_id=model_id,
        input_text=prompt,
        output_text=output,
        latency_ms=latency_ms,
    )
    return {"verification_result": verification_result, "metadata": metadata}
