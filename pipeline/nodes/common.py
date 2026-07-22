from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from config.models import build_chat_model, estimate_cost_usd
from pipeline.state import RAGState


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def load_prompt(domain: str, role: str) -> str:
    path = PROMPT_ROOT / domain / f"{role}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return (PROMPT_ROOT / "cs" / f"{role}.txt").read_text(encoding="utf-8")


def render_prompt(template: str, **values: Any) -> str:
    safe_values = {
        key: json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list)) else value
        for key, value in values.items()
    }
    return template.format(**safe_values)


def add_node_metadata(
    state: RAGState,
    node_name: str,
    *,
    model_id: str | None,
    input_text: str,
    output_text: str,
    latency_ms: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    cost = estimate_cost_usd(model_id, input_tokens, output_tokens) if model_id else 0.0
    metadata = dict(state.get("metadata", {}))
    metadata[node_name] = {
        "model": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "latency_ms": latency_ms,
        **(extra or {}),
    }
    return metadata


def invoke_llm(model_id: str, prompt: str) -> tuple[str, int]:
    model = build_chat_model(model_id)
    start = time.perf_counter()
    response = model.invoke(prompt)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return str(response.content).strip(), latency_ms
