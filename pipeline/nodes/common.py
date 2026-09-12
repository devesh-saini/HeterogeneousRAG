from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from config.models import (
    GROQ_GPT_OSS_REASONING_EFFORT,
    build_chat_model,
    estimate_cost_usd,
    parse_model_spec,
)
from pipeline.rate_limit import (
    cancel_groq_reservation,
    complete_groq_reservation,
    get_groq_rate_limit_settings,
    is_daily_rate_limit_error,
    is_rate_limit_error,
    reserve_groq_tokens,
    retry_after_seconds,
)
from pipeline.state import RAGState


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def estimate_provider_tokens(text: str) -> int:
    """Conservative tokenizer-free estimate used only for pre-request pacing."""
    words = len(re.findall(r"\S+", text))
    characters = len(text)
    return max(1, math.ceil(words * 1.35), math.ceil(characters / 3.5)) + 32


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
    details = dict(extra or {})
    input_tokens = int(details.pop("input_tokens", estimate_tokens(input_text)))
    output_tokens = int(details.pop("output_tokens", estimate_tokens(output_text)))
    cost = estimate_cost_usd(model_id, input_tokens, output_tokens) if model_id else 0.0
    existing = state.get("metadata", {})
    if "events" in existing:
        events = list(existing.get("events", []))
        latest = dict(existing.get("latest", {}))
    else:
        # Upgrade legacy in-memory metadata shape without losing entries.
        events = [
            {"node": name, "attempt": 0, **details}
            for name, details in existing.items()
            if isinstance(details, dict)
        ]
        latest = {
            name: details for name, details in existing.items() if isinstance(details, dict)
        }
    event = {
        "node": node_name,
        "attempt": int(state.get("retry_count", 0)),
        "model": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "latency_ms": latency_ms,
        **details,
    }
    events.append(event)
    latest[node_name] = event
    return {"events": events, "latest": latest}


def _response_token_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    provider_usage = response_metadata.get("token_usage", {})
    input_tokens = usage.get("input_tokens", provider_usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", provider_usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens", provider_usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)
    return (
        int(input_tokens) if input_tokens is not None else None,
        int(output_tokens) if output_tokens is not None else None,
        int(total_tokens) if total_tokens is not None else None,
    )


def invoke_llm(
    model_id: str, prompt: str, *, max_tokens: int
) -> tuple[str, int, dict[str, Any]]:
    spec = parse_model_spec(model_id)
    model = build_chat_model(model_id, max_tokens=max_tokens)
    estimated_input_tokens = estimate_provider_tokens(prompt)
    estimated_total_tokens = estimated_input_tokens + max_tokens
    quota_wait_ms = 0
    service_latency_ms = 0
    rate_limit_retries = 0

    settings = get_groq_rate_limit_settings()
    if (
        spec.provider == "groq"
        and settings.enabled
        and estimated_total_tokens > settings.working_token_budget
    ):
        raise ValueError(
            f"Estimated Groq request size ({estimated_total_tokens} tokens) exceeds "
            f"the configured working budget ({settings.working_token_budget}). "
            "Reduce the prompt/context or the role's output cap."
        )

    while True:
        reservation = None
        if spec.provider == "groq":
            reservation, wait_ms = reserve_groq_tokens(estimated_total_tokens)
            quota_wait_ms += wait_ms

        start = time.perf_counter()
        try:
            response = model.invoke(prompt)
        except Exception as error:
            service_latency_ms += int((time.perf_counter() - start) * 1_000)
            cancel_groq_reservation(reservation)
            settings = get_groq_rate_limit_settings()
            if (
                spec.provider == "groq"
                and is_rate_limit_error(error)
                and is_daily_rate_limit_error(error)
            ):
                raise RuntimeError(
                    "Groq's daily free-tier limit has been reached. Completed questions "
                    "are already stored; resume this experiment later with --resume. "
                    f"Groq response: {error}"
                ) from error
            if (
                spec.provider != "groq"
                or not is_rate_limit_error(error)
                or rate_limit_retries >= settings.max_429_retries
            ):
                raise
            rate_limit_retries += 1
            retry_wait = retry_after_seconds(error, rate_limit_retries)
            print(
                "Groq returned HTTP 429; "
                f"waiting {retry_wait:.1f}s before retry "
                f"{rate_limit_retries}/{settings.max_429_retries}.",
                flush=True,
            )
            time.sleep(retry_wait)
            quota_wait_ms += round(retry_wait * 1_000)
            continue

        service_latency_ms += int((time.perf_counter() - start) * 1_000)
        input_tokens, output_tokens, total_tokens = _response_token_usage(response)
        if spec.provider == "groq":
            complete_groq_reservation(reservation, total_tokens)
        break

    usage_source = "provider" if total_tokens is not None else "estimated"
    final_input_tokens = input_tokens or estimated_input_tokens
    final_output_tokens = output_tokens or estimate_tokens(str(response.content))
    final_total_tokens = total_tokens or final_input_tokens + final_output_tokens
    return (
        str(response.content).strip(),
        service_latency_ms,
        {
            "input_tokens": final_input_tokens,
            "output_tokens": final_output_tokens,
            "total_tokens": final_total_tokens,
            "token_count_source": usage_source,
            "estimated_input_tokens": estimated_input_tokens,
            "reserved_tokens": estimated_total_tokens,
            "max_output_tokens": max_tokens,
            "reasoning_effort": (
                GROQ_GPT_OSS_REASONING_EFFORT
                if spec.provider == "groq" and spec.model.startswith("openai/gpt-oss-")
                else None
            ),
            "rate_limit_wait_ms": quota_wait_ms,
            "rate_limit_retries": rate_limit_retries,
        },
    )
