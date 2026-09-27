import json
from pathlib import Path

import pytest

from laya_bangla.data import load_examples, load_inputs, load_questions

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


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ('revision = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"', "revision = 123", "revision"),
        ('dataset = "../examples/bangla_smoke.jsonl"', "dataset = 123", "dataset"),
        ('questions_bn = "questions.bn.json"', 'questions_bn = ""', "questions_bn"),
        ("seed = 42", "seed = 4294967296", "seed"),
    ],
)
def test_bad_config_types_and_seed_fail_before_loading_data(
    tmp_path, original, replacement, message
):
    content = (ROOT / "configs/baseline.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(content.replace(original, replacement), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_inputs(path)


def test_duplicate_question_keys_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        '{"task":{"type":"choice","instructions":"Route?",'
        '"criteria":{"A":"First","B":"Second","A":"Overwritten"}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate.*A"):
        load_questions(path)


def test_duplicate_reference_labels_are_rejected(tmp_path):
    _, schemas, _ = load_inputs(ROOT / "configs/baseline.toml")
    path = tmp_path / "examples.jsonl"
    path.write_text(
        '{"id":"a","group_id":"g","slice":"bangla","state":"বাংলা",'
        '"expected":{"department":"billing","churn_risk":true,"churn_risk":false,"urgency":0}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate.*churn_risk"):
        load_examples(path, schemas["bn"])
