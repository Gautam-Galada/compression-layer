"""Cheap deterministic scoring helpers."""

from collections import Counter


def normalize_answer(text: str) -> str:
    return " ".join(text.lower().split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)

    overlap = Counter(prediction_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def code_reference_match(prediction: str, gold: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)

    if normalized_prediction and normalized_gold and normalized_gold in normalized_prediction:
        return 1.0

    return token_f1(prediction, gold)


def code_reference_exact_match(prediction: str, gold: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    return float(
        bool(normalized_prediction)
        and bool(normalized_gold)
        and normalized_gold in normalized_prediction
    )
