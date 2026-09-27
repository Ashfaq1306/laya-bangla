import math

import pytest

from laya_bangla.metrics import decision_record, summarize


def test_binary_probability_is_not_entropy_confidence():
    q = {"type": "noul"}
    row = decision_record({"type": "noul", "noul": 0.8, "confidence": 0.01}, q, True)
    assert row["confidence"] == 0.8
    assert row["brier"] == pytest.approx(0.08)  # sum over BOTH labels
    assert row["nll"] == pytest.approx(-math.log(0.8))
    wrong = decision_record({"type": "noul", "noul": 0.8}, q, False)
    metrics = summarize([row, wrong], ["false", "true"], bins=10)
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["ece"] == pytest.approx(0.3)


def test_score_mae_uses_expected_score_not_argmax():
    question = {"type": "score", "criteria": ["low", "medium", "high"]}
    row = decision_record(
        {"type": "score", "score": 0.9, "probabilities": {"0": 0.4, "1": 0.3, "2": 0.3}},
        question,
        1,
    )
    assert row["predicted"] == "0"
    assert row["score_absolute_error"] == pytest.approx(0.1)
    assert summarize([row], ["0", "1", "2"])["score_mae"] == pytest.approx(0.1)


def test_rounding_ties_keep_the_reported_choice():
    q = {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}}
    row = decision_record(
        {"type": "choice", "choice": "b", "probabilities": {"a": 0.3333, "b": 0.3333, "c": 0.3333}},
        q,
        "b",
    )
    assert row["correct"]
    assert row["confidence"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probabilities_are_not_silently_scored(probability):
    with pytest.raises(ValueError, match="finite"):
        decision_record({"type": "noul", "noul": probability}, {"type": "noul"}, True)


def test_probability_one_is_included_in_ece():
    row = decision_record({"type": "noul", "noul": 1.0}, {"type": "noul"}, False)
    metrics = summarize([row], ["false", "true"])
    assert metrics["ece"] == 1.0
    assert metrics["brier"] == 2.0
    assert math.isfinite(metrics["nll"])
