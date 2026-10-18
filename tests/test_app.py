"""The playground API with fake agents; skipped when the app extra is not installed."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from laya_bangla.app import PRESETS, create_app  # noqa: E402

from .test_benchmark import FakeAgent  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    models = {"fine-tuned": FakeAgent(scenario="alarm"), "base": FakeAgent(scenario="news")}
    app = create_app(
        models,
        {"fine-tuned": "local", "base": "upstream"},
        ROOT / "configs/tasks.json",
        ROOT / "configs/baseline.toml",
    )
    return TestClient(app)


def test_index_and_info(client):
    assert "laya-bangla playground" in client.get("/").text
    info = client.get("/api/info").json()
    assert info["models"] == {"fine-tuned": "local", "base": "upstream"}
    assert set(info["presets"]) == set(PRESETS) | {"custom"}


@pytest.mark.parametrize("preset", sorted(PRESETS))
@pytest.mark.parametrize("lang", ["bn", "en"])
def test_every_preset_answers_with_both_models(client, preset, lang):
    response = client.post(
        "/api/predict", json={"text": "আমার বিল ভুল", "preset": preset, "lang": lang}
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data["results"]) == {"fine-tuned", "base"}
    for result in data["results"].values():
        assert set(data["questions"]) <= set(result["answers"])


def test_assistant_preset_routes_to_an_intent_in_the_predicted_scenario(client):
    data = client.post("/api/predict", json={"text": "সাতটায় জাগাও", "preset": "assistant"}).json()
    assert data["results"]["fine-tuned"]["answers"]["massive_intent"]["choice"].startswith("alarm_")
    # A single-intent scenario has no second stage.
    assert "massive_intent" not in data["results"]["base"]["answers"]


def test_custom_questions_and_validation_errors(client):
    questions = {
        "spam": {
            "type": "noul",
            "instructions": "Is this spam?",
            "criteria": {"false": "no", "true": "yes"},
        }
    }
    data = client.post(
        "/api/predict", json={"text": "free offer", "preset": "custom", "questions": questions}
    ).json()
    assert data["results"]["base"]["answers"]["spam"]["noul"] == 0.25
    assert client.post("/api/predict", json={"text": "x", "preset": "custom"}).status_code == 422
    assert client.post("/api/predict", json={"text": "x", "preset": "nope"}).status_code == 422
    assert client.post("/api/predict", json={"text": "", "preset": "support"}).status_code == 422
    assert client.get("/api/presets/nope").status_code == 404
