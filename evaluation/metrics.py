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
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


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
