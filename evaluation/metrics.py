from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


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
        equal = float(pred_tokens == gold_tokens)
        return {"precision": equal, "recall": equal, "f1": equal}
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def score_answer(prediction: str, references: list[str]) -> dict[str, float | int]:
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
    for index, reference in enumerate(valid_references):
        scores = token_scores(prediction, reference)
        normalized_reference = normalize_text(reference)
        reference_length = len(normalized_reference.split())
        candidates.append(
            {
                **scores,
                "reference_index": index,
                "exact_match": exact_match(prediction, reference),
                "reference_contained": float(
                    bool(normalized_reference)
                    and normalized_reference in normalized_prediction
                ),
                "verbosity_ratio": (
                    prediction_length / reference_length if reference_length else 0.0
                ),
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
    return {
        "exact_match": max(float(item["exact_match"]) for item in candidates),
        "f1": float(best["f1"]),
        "answer_precision": float(best["precision"]),
        "answer_recall": max(float(item["recall"]) for item in candidates),
        "reference_contained": max(
            float(item["reference_contained"]) for item in candidates
        ),
        "verbosity_ratio": float(best["verbosity_ratio"]),
        "matched_reference_index": int(best["reference_index"]),
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
