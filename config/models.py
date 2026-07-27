from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


RoleName = Literal["rewriter", "retriever", "reranker", "synthesizer", "verifier"]
ConfigName = Literal["homogeneous", "heterogeneous"]


HOMOGENEOUS_CONFIG: dict[RoleName, str] = {
    "rewriter": "groq/openai/gpt-oss-20b",
    "retriever": "groq/openai/gpt-oss-20b",
    "reranker": "groq/openai/gpt-oss-20b",
    "synthesizer": "groq/openai/gpt-oss-20b",
    "verifier": "groq/openai/gpt-oss-20b",
}

HETEROGENEOUS_CONFIG: dict[RoleName, str] = {
    "rewriter": "ollama/qwen2.5:7b",
    "retriever": "ollama/qwen2.5:7b",
    "reranker": "ollama/qwen2.5:7b",
    "synthesizer": "groq/llama-3.1-70b-versatile",
    "verifier": "groq/llama-3.1-70b-versatile",
}

MODEL_CONFIGS: dict[ConfigName, dict[RoleName, str]] = {
    "homogeneous": HOMOGENEOUS_CONFIG,
    "heterogeneous": HETEROGENEOUS_CONFIG,
}

# Fill these with your actual Groq pricing if you want dollar-accurate accounting.
# Values are USD per 1M tokens.
MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "ollama/qwen2.5:7b": {"input": 0.0, "output": 0.0},
    "groq/llama-3.1-8b-instant": {"input": 0.0, "output": 0.0},
    "groq/llama-3.1-70b-versatile": {"input": 0.0, "output": 0.0},
    "groq/qwen2.5-72b": {"input": 0.0, "output": 0.0},
}


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    raw: str


def parse_model_spec(model_id: str) -> ModelSpec:
    provider, model = model_id.split("/", 1)
    return ModelSpec(provider=provider, model=model, raw=model_id)


def get_model_config(config_name: str) -> dict[RoleName, str]:
    try:
        return MODEL_CONFIGS[config_name]  # type: ignore[index]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown config '{config_name}'. Expected one of: {valid}") from exc


def build_chat_model(model_id: str, temperature: float = 0.0) -> "BaseChatModel":
    spec = parse_model_spec(model_id)
    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=spec.model, temperature=temperature)
    if spec.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=spec.model, temperature=temperature)
    raise ValueError(f"Unsupported model provider '{spec.provider}' in '{model_id}'")


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING_USD_PER_1M.get(model_id, {"input": 0.0, "output": 0.0})
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )
