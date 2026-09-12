from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


RoleName = Literal["rewriter", "retriever", "reranker", "synthesizer", "verifier"]
ConfigName = Literal["homogeneous", "heterogeneous"]


HOMOGENEOUS_CONFIG: dict[RoleName, str] = {
    "rewriter": "ollama/qwen2.5:latest",
    "retriever": "ollama/qwen2.5:latest",
    "reranker": "ollama/qwen2.5:latest",
    "synthesizer": "ollama/qwen2.5:latest",
    "verifier": "ollama/qwen2.5:latest",
}

HETEROGENEOUS_CONFIG: dict[RoleName, str] = {
    "rewriter": "ollama/qwen2.5:latest",
    "retriever": "ollama/qwen2.5:latest",
    "reranker": "groq/openai/gpt-oss-20b",
    "synthesizer": "groq/openai/gpt-oss-120b",
    "verifier": "groq/openai/gpt-oss-120b",
}

MODEL_CONFIGS: dict[ConfigName, dict[RoleName, str]] = {
    "homogeneous": HOMOGENEOUS_CONFIG,
    "heterogeneous": HETEROGENEOUS_CONFIG,
}

# Output caps bound the completion side of free-tier Groq usage and make the
# generation budget identical for a role across configurations.
ROLE_MAX_OUTPUT_TOKENS: dict[RoleName, int] = {
    "rewriter": 256,
    "retriever": 1,
    "reranker": 512,
    "synthesizer": 768,
    "verifier": 768,
}

GROQ_GPT_OSS_REASONING_EFFORT = "low"
RERANKER_PROMPT_CHARACTER_BUDGET = 18_000

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


def build_chat_model(
    model_id: str, temperature: float = 0.0, max_tokens: int | None = None
) -> "BaseChatModel":
    spec = parse_model_spec(model_id)
    if spec.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=spec.model,
            temperature=temperature,
            num_predict=max_tokens,
        )
    if spec.provider == "groq":
        from langchain_groq import ChatGroq

        # Retries are handled in the shared invocation layer so their waiting
        # time and count are visible in experiment metadata.
        groq_options: dict[str, object] = {}
        if spec.model.startswith("openai/gpt-oss-"):
            # GPT-OSS defaults to medium reasoning. Low effort preserves a
            # reasoning stage while keeping free-tier completion usage bounded.
            groq_options["reasoning_effort"] = GROQ_GPT_OSS_REASONING_EFFORT
        return ChatGroq(
            model=spec.model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=0,
            **groq_options,
        )
    raise ValueError(f"Unsupported model provider '{spec.provider}' in '{model_id}'")


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING_USD_PER_1M.get(model_id, {"input": 0.0, "output": 0.0})
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )
