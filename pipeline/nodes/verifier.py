from __future__ import annotations

import json
import re

from config.models import ROLE_MAX_OUTPUT_TOKENS
from pipeline.nodes.common import add_node_metadata, invoke_llm, load_prompt, render_prompt
from pipeline.state import RAGState


def _extract_json(output: str) -> object:
    text = output.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() == "true":
            return True
        if value.strip().lower() == "false":
            return False
    return default


def _parse_verification(output: str) -> dict[str, object]:
    try:
        parsed = _extract_json(output)
        if not isinstance(parsed, dict):
            raise ValueError("Verifier JSON must be an object")
        grounded = _as_bool(
            parsed.get("groundedness_verdict", parsed.get("grounded", parsed.get("verdict")))
        )
        relevant = _as_bool(
            parsed.get("relevance_verdict", parsed.get("relevant", parsed.get("verdict")))
        )
        complete = _as_bool(
            parsed.get("completeness_verdict", parsed.get("complete", parsed.get("verdict")))
        )
        return {
            "parse_success": True,
            "verdict": grounded and relevant and complete,
            "groundedness_verdict": grounded,
            "relevance_verdict": relevant,
            "completeness_verdict": complete,
            "reasoning": str(parsed.get("reasoning", "")),
            "unsupported_claims": [
                str(item) for item in parsed.get("unsupported_claims", [])
            ]
            if isinstance(parsed.get("unsupported_claims", []), list)
            else [],
        }
    except Exception:
        return {
            "parse_success": False,
            "verdict": False,
            "groundedness_verdict": False,
            "relevance_verdict": False,
            "completeness_verdict": False,
            "reasoning": f"Verifier returned invalid JSON: {output[:500]}",
            "unsupported_claims": [],
        }


def verifier_node(state: RAGState) -> RAGState:
    model_id = state["model_assignment"]["verifier"]
    prompt = render_prompt(
        load_prompt(state["domain"], "verifier"),
        original_query=state["original_query"],
        answer_type=state.get("answer_type", "unknown"),
        answer=state.get("synthesized_answer", ""),
        evidence=state.get("reranked_chunks", []),
    )
    output, latency_ms, invocation_metadata = invoke_llm(
        model_id,
        prompt,
        max_tokens=ROLE_MAX_OUTPUT_TOKENS["verifier"],
    )
    verification_result = _parse_verification(output)
    metadata = add_node_metadata(
        state,
        "verifier",
        model_id=model_id,
        input_text=prompt,
        output_text=output,
        latency_ms=latency_ms,
        extra=invocation_metadata,
    )
    return {"verification_result": verification_result, "metadata": metadata}
