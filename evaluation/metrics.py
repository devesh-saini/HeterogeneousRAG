from __future__ import annotations

import re
import string
from collections import Counter


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


def retrieval_recall(retrieved_chunks: list[str], gold_evidence: list[str]) -> float:
    if not gold_evidence:
        return 0.0
    retrieved = normalize_text(" ".join(retrieved_chunks))
    hits = sum(1 for evidence in gold_evidence if normalize_text(evidence) in retrieved)
    return hits / len(gold_evidence)
