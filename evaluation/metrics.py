from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol


class SentenceEncoder(Protocol):
    def encode(self, sentences: list[str], *, normalize_embeddings: bool) -> Any: ...


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_text(prediction) == normalize_text(gold))


def token_f1(prediction: str, gold: str) -> float:
    return token_scores(prediction, gold)["f1"]


def token_scores(prediction: str, gold: str) -> dict[str, float]:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        # Match the official QASPER evaluator: empty token sequences receive 0.
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0}
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0}
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
        # F2 weights coverage twice as strongly as precision. It complements,
        # rather than replaces, the official QASPER/SQuAD-style token F1.
        "f2": 5 * precision * recall / (4 * precision + recall),
    }


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    """Return LCS length using O(min(len(left), len(right))) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_scores(prediction: str, gold: str) -> dict[str, float]:
    prediction_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not prediction_tokens or not gold_tokens:
        equal = float(prediction_tokens == gold_tokens)
        return {"precision": equal, "recall": equal, "f1": equal}
    common = _lcs_length(prediction_tokens, gold_tokens)
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _categorical_value(text: str, answer_type: str) -> str | None:
    normalized = normalize_text(text)
    if answer_type in {"yes_no", "boolean"}:
        match = re.match(r"^(yes|true|no|false)\b", normalized)
        if not match:
            return None
        return "yes" if match.group(1) in {"yes", "true"} else "no"
    if answer_type in {"unanswerable", "none"}:
        aliases = (
            "unanswerable",
            "insufficient evidence",
            "insufficient information",
            "not enough information",
            "cannot be answered",
        )
        return "unanswerable" if any(normalized.startswith(alias) for alias in aliases) else None
    return None


def semantic_similarity(
    prediction: str,
    references: list[str],
    encoder: SentenceEncoder,
) -> float:
    """Maximum cosine similarity using normalized sentence embeddings.

    This is diagnostic because embedding similarity is not the official dataset
    metric. It is inexpensive when the retrieval embedder is already in memory.
    """
    if not prediction.strip() or not references:
        return 0.0
    embeddings = encoder.encode(
        [prediction, *references], normalize_embeddings=True
    )
    prediction_embedding = embeddings[0]
    similarities = [
        float(sum(float(x) * float(y) for x, y in zip(prediction_embedding, reference)))
        for reference in embeddings[1:]
    ]
    return max(0.0, min(1.0, max(similarities, default=0.0)))


def score_answer(
    prediction: str,
    references: list[str],
    answer_type: str = "unknown",
    reference_types: list[str] | None = None,
    semantic_encoder: SentenceEncoder | None = None,
) -> dict[str, float | int | None]:
    """Score against all valid references and expose verbosity separately.

    F1 remains available for comparability, while reference recall and containment
    reveal answers that contain the expected content but are unnecessarily verbose.
    """
    valid_references = [reference for reference in references if normalize_text(reference)]
    if not valid_references:
        valid_references = [""]

    candidates: list[dict[str, float | int]] = []
    normalized_prediction = normalize_text(prediction)
    prediction_length = len(normalized_prediction.split())
    valid_reference_types = (
        reference_types
        if reference_types is not None and len(reference_types) == len(valid_references)
        else [answer_type] * len(valid_references)
    )
    for index, (reference, reference_type) in enumerate(
        zip(valid_references, valid_reference_types)
    ):
        scores = token_scores(prediction, reference)
        rouge_l = rouge_l_scores(prediction, reference)
        normalized_reference = normalize_text(reference)
        reference_length = len(normalized_reference.split())
        candidates.append(
            {
                **scores,
                "reference_index": index,
                "answer_type": reference_type,
                "exact_match": exact_match(prediction, reference),
                "reference_contained": float(
                    bool(normalized_reference)
                    and normalized_reference in normalized_prediction
                ),
                "verbosity_ratio": (
                    prediction_length / reference_length if reference_length else 0.0
                ),
                "rouge_l_f1": rouge_l["f1"],
            }
        )

    best = max(
        candidates,
        key=lambda item: (
            float(item["f1"]),
            float(item["recall"]),
            float(item["precision"]),
        ),
    )
    categorical_matches: list[bool] = []
    for reference, reference_type in zip(valid_references, valid_reference_types):
        categorical_reference = _categorical_value(reference, reference_type)
        if categorical_reference is not None:
            categorical_matches.append(
                _categorical_value(prediction, reference_type) == categorical_reference
            )
    type_accuracy = float(any(categorical_matches)) if categorical_matches else None
    semantic_score = (
        semantic_similarity(prediction, valid_references, semantic_encoder)
        if semantic_encoder is not None
        else None
    )
    return {
        # This is the official-compatible primary answer metric: maximum
        # SQuAD-style token F1 over human references.
        "exact_match": max(float(item["exact_match"]) for item in candidates),
        "f1": float(best["f1"]),
        "f2": max(float(item["f2"]) for item in candidates),
        "rouge_l_f1": max(float(item["rouge_l_f1"]) for item in candidates),
        "semantic_similarity": semantic_score,
        "type_accuracy": type_accuracy,
        "answer_precision": float(best["precision"]),
        "answer_recall": max(float(item["recall"]) for item in candidates),
        "reference_contained": max(
            float(item["reference_contained"]) for item in candidates
        ),
        "verbosity_ratio": float(best["verbosity_ratio"]),
        "matched_reference_index": int(best["reference_index"]),
        "matched_answer_type": str(best["answer_type"]),
        "reference_count": len(valid_references),
    }


def token_recall(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    return sum(overlap.values()) / len(gold_tokens)


def _looks_like_id(value: str) -> bool:
    return bool(re.fullmatch(r"\d+|[a-f0-9]{20,}", value.strip().lower()))


def _metadata_source_ids(retrieved_metadata: list[dict[str, Any]] | None) -> set[str]:
    source_ids: set[str] = set()
    for metadata in retrieved_metadata or []:
        for key in ("source_id", "passage_id", "id"):
            value = metadata.get(key)
            if value not in (None, ""):
                source_ids.add(str(value))
    return source_ids


def retrieval_recall(
    retrieved_chunks: list[str],
    gold_evidence: list[str],
    retrieved_metadata: list[dict[str, Any]] | None = None,
) -> float:
    if not gold_evidence:
        return 0.0
    source_ids = _metadata_source_ids(retrieved_metadata)
    if source_ids and all(_looks_like_id(str(evidence)) for evidence in gold_evidence):
        hits = sum(1 for evidence in gold_evidence if str(evidence) in source_ids)
        return hits / len(gold_evidence)

    retrieved = normalize_text(" ".join(retrieved_chunks))
    if not retrieved:
        return 0.0
    hits = 0
    for evidence in gold_evidence:
        normalized_evidence = normalize_text(evidence)
        if not normalized_evidence:
            continue
        if normalized_evidence in retrieved or token_recall(retrieved, evidence) >= 0.6:
            hits += 1
    return hits / len(gold_evidence)
