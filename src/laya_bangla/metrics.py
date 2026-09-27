"""Metrics derived from typed answers, never from Laya's entropy confidence."""

import math
from collections import Counter

from .data import label_space


def decision_record(answer: dict, question: dict, expected) -> dict:
    kind = question["type"]
    if answer.get("type") != kind:
        raise ValueError(f"Answer type does not match {kind}")
    labels = label_space(question)
    if kind == "noul":
        probability = float(answer["noul"])
        probabilities = {"false": 1 - probability, "true": probability}
        target = str(expected).lower()
        predicted = "true" if probability >= 0.5 else "false"
    else:
        probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
        if set(probabilities) != set(labels):
            raise ValueError("Answer probabilities must cover the declared labels exactly")
        target = str(expected)
        predicted = (
            answer["choice"] if kind == "choice" else max(labels, key=probabilities.__getitem__)
        )
    if target not in labels or predicted not in labels:
        raise ValueError("Target or predicted label is outside the schema")
    values = list(probabilities.values())
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    total = sum(values)
    if not math.isclose(total, 1.0, abs_tol=0.001):
        raise ValueError("Probabilities must sum to 1 (allowing SDK rounding)")
    # The SDK rounds probabilities to four decimal places. Restore unit mass.
    probabilities = {label: probabilities[label] / total for label in labels}
    if probabilities[predicted] + 0.0002 < max(probabilities.values()):
        raise ValueError("Reported choice disagrees with the probability distribution")
    score_error = None
    if kind == "score":
        score = float(answer["score"])
        if not math.isfinite(score) or not 0 <= score <= len(labels) - 1:
            raise ValueError("Ordinal score is outside the declared scale")
        score_error = abs(score - expected)
    return {
        "expected": target,
        "predicted": predicted,
        "correct": predicted == target,
        "confidence": max(probabilities.values()),
        "brier": sum((p - int(label == target)) ** 2 for label, p in probabilities.items()),
        "nll": -math.log(max(probabilities[target], 1e-12)),
        "score_absolute_error": score_error,
    }


def summarize(records: list[dict], labels: list[str], bins: int = 10) -> dict:
    if not records or bins <= 0:
        raise ValueError("Metrics require records and a positive bin count")
    n = len(records)
    expected_counts = Counter(row["expected"] for row in records)
    predicted_counts = Counter(row["predicted"] for row in records)
    confusion = {target: {pred: 0 for pred in labels} for target in labels}
    buckets = [[] for _ in range(bins)]
    for row in records:
        confusion[row["expected"]][row["predicted"]] += 1
        buckets[min(int(row["confidence"] * bins), bins - 1)].append(row)
    f1 = []
    for label in labels:
        tp = confusion[label][label]
        denominator = expected_counts[label] + predicted_counts[label]
        f1.append(2 * tp / denominator if denominator else 0.0)
    ece = sum(
        abs(sum(row["confidence"] for row in bucket) - sum(row["correct"] for row in bucket)) / n
        for bucket in buckets
        if bucket
    )
    score_errors = [row["score_absolute_error"] for row in records]
    return {
        "n": n,
        "accuracy": sum(row["correct"] for row in records) / n,
        "macro_f1": sum(f1) / len(labels),
        "majority_accuracy": max(expected_counts.values()) / n,
        "brier": sum(row["brier"] for row in records) / n,
        "nll": sum(row["nll"] for row in records) / n,
        "ece": ece,
        "score_mae": sum(score_errors) / n if all(x is not None for x in score_errors) else None,
        "confusion_matrix": confusion,
    }
