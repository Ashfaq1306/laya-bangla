import json
from pathlib import Path

import pytest

from laya_bangla.data import load_examples, load_inputs

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_set_is_balanced_and_grouped():
    _, schemas, examples = load_inputs(ROOT / "configs/baseline.toml")
    assert len(examples) == 36
    groups = {}
    for example in examples:
        groups.setdefault(example["group_id"], set()).add(example["slice"])
    assert len(groups) == 12
    assert all(slices == {"bangla", "banglish", "mixed"} for slices in groups.values())
    assert {q["type"] for q in schemas["bn"].values()} == {"choice", "noul", "score"}
    for name in ("bangla", "banglish", "mixed"):
        rows = [item for item in examples if item["slice"] == name]
        assert sum(item["expected"]["churn_risk"] for item in rows) == 6
        assert all(sum(item["expected"]["urgency"] == n for item in rows) == 4 for n in range(3))


@pytest.mark.parametrize(
    "failure", ["duplicate", "boolean_score", "group_disagreement", "missing_label"]
)
def test_invalid_examples_fail_before_inference(tmp_path, failure):
    _, schemas, examples = load_inputs(ROOT / "configs/baseline.toml")
    rows = examples[:2]
    if failure == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif failure == "boolean_score":
        rows[0]["expected"]["urgency"] = True
    elif failure == "group_disagreement":
        rows[1]["expected"]["department"] = "other"
    else:
        del rows[0]["expected"]["churn_risk"]
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_examples(path, schemas["bn"])


def test_mutable_revision_is_rejected(tmp_path):
    path = tmp_path / "baseline.toml"
    path.write_text('revision = "main"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        load_inputs(path)
