"""The Space asks the same questions as the benchmark and the local playground."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from laya_bangla import app
from laya_bangla.data import load_questions
from laya_bangla.prepare import load_tasks, render_question

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playground", ROOT / "space/playground.py")
playground = importlib.util.module_from_spec(spec)
spec.loader.exec_module(playground)
TASKS = load_tasks(ROOT / "configs/tasks.json")
SUPPORT = playground.load_support_questions()


def test_support_questions_match_the_diagnostic_schemas():
    for lang in ("bn", "en"):
        assert SUPPORT[lang] == load_questions(ROOT / f"configs/questions.{lang}.json")


@pytest.mark.parametrize("preset", sorted(app.PRESETS))
@pytest.mark.parametrize("lang", ["bn", "en"])
def test_presets_match_the_local_playground(preset, lang):
    assert playground.PRESETS == app.PRESETS
    expected = app.preset_questions(TASKS, SUPPORT, preset, lang)
    assert playground.preset_questions(TASKS, SUPPORT, preset, lang) == expected
    assert playground.parse_questions(json.dumps(expected)) == expected


def test_intent_follow_up_uses_the_scenario_options():
    task = TASKS["massive_intent"]
    question = playground.intent_question(TASKS, "alarm", "bn")["massive_intent"]
    assert question == render_question(
        task, task["instructions"]["eval"]["bn"], "bn", task["groups"]["alarm"]
    )
    assert playground.intent_question(TASKS, "weather", "en") is None


def test_examples_and_custom_template_are_valid():
    assert set(playground.PRESET_LABELS) == set(app.PRESETS) | {"custom"}
    assert {preset for _text, preset in playground.EXAMPLES} <= set(app.PRESETS)
    template = json.dumps(playground.CUSTOM_TEMPLATE, ensure_ascii=False)
    assert playground.parse_questions(template) == playground.CUSTOM_TEMPLATE


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{", "not valid JSON"),
        ("[]", "JSON object"),
        ('{"q": {"type": "rank", "instructions": "x", "criteria": {}}}', "type must be"),
        ('{"q": {"type": "choice", "instructions": " ", "criteria": {"a": "", "b": ""}}}', "instr"),
        ('{"q": {"type": "choice", "instructions": "x", "criteria": {"a": "only"}}}', "at least 2"),
        ('{"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}', "at least 2"),
        ('{"q": {"type": "noul", "instructions": "x", "criteria": {"no": "", "yes": ""}}}', "true"),
    ],
)
def test_parse_questions_rejects_invalid_schemas(text, message):
    with pytest.raises(ValueError, match=message):
        playground.parse_questions(text)


def test_result_card_shows_every_option_and_escapes_text():
    questions = {
        "c": {"type": "choice", "instructions": "<b>x</b>", "criteria": {"a": "A", "b": "B"}},
        "n": {"type": "noul", "instructions": "y", "criteria": {"false": "no", "true": "yes"}},
        "s": {"type": "score", "instructions": "z", "criteria": ["low", "mid", "high"]},
    }
    answers = {
        "c": {"type": "choice", "choice": "b", "probabilities": {"a": 0.3, "b": 0.7}},
        "n": {"type": "noul", "noul": 0.2},
        "s": {"type": "score", "score": 1.2, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}},
    }
    assert playground.answer_rows(answers["n"], questions["n"])[0] == "No / না"
    assert playground.answer_rows(answers["s"], questions["s"])[0] == "Level 1: mid"
    card = playground.result_card("m", "sub", answers, questions, 12.0)
    assert "&lt;b&gt;x&lt;/b&gt;" in card and "<b>x</b>" not in card
    assert card.count('class="lb-row"') == 7
    assert "70.0%" in card and "80.0%" in card and "60.0%" in card


def summary(accuracy):
    def m(n):
        return {"all": {"n": n, "accuracy": accuracy}}

    return {
        "tasks": {
            "banth/hate": {"bn": m(100), "en": m(100)},
            "indicxnli/entailment": {"bn": m(50), "en": m(50)},
        },
        "massive_end_to_end_intent": {"bn": {"n": 10, "accuracy": accuracy}},
    }


def test_benchmark_table_bolds_the_better_model_and_pools_trained_sources():
    base, tuned = summary(0.5), summary(0.75)
    tuned["tasks"]["indicxnli/entailment"]["bn"]["all"]["accuracy"] = 0.25
    tuned["tasks"]["indicxnli/entailment"]["en"]["all"]["accuracy"] = 0.25
    rows = [
        ("banth/hate", "Hate", "BanTH", "Banglish"),
        ("indicxnli/entailment", "NLI", "IndicXNLI", "Bangla"),
    ]
    table = playground.benchmark_table(base, tuned, rows).splitlines()
    assert table[2] == "| Hate | BanTH | Banglish | 100 | 50% / 50% | **75% / 75%** |"
    assert table[3] == "| NLI | IndicXNLI | Bangla | 50 | **50% / 50%** | 25% / 25% |"
    assert playground.pooled_accuracy(tuned, ("banth",)) == 0.75


def test_space_card_pins_the_benchmark_inference_stack():
    readme = (ROOT / "space/README.md").read_text(encoding="utf-8")
    requirements = (ROOT / "space/requirements.txt").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    for package in ("laya", "transformers", "huggingface-hub", "numpy"):
        pin = re.search(rf"^{package}==(\S+)$", requirements, re.M).group(1)
        assert f'name = "{package}"\nversion = "{pin}"' in lock
    assert "torch==2.8.0+cpu" in requirements
    short = re.search(r"^short_description: (.+)$", readme, re.M).group(1)
    assert len(short) <= 60
