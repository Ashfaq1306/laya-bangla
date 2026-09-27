"""Exercise run artifacts and failure handling without downloading model weights."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from laya_bangla import cli
from laya_bangla.data import load_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_context(tmp_path, monkeypatch):
    config_path = ROOT / "configs/baseline.toml"
    config, schemas, examples = load_inputs(config_path)
    snapshot = tmp_path / config["revision"]
    for name in cli.CHECKPOINT_FILES:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    class Agent:
        device = "cpu"
        cfg = {"temperature": [1.0, 1.0, 1.0]}
        dtype = "float32"
        amp_enabled = False
        mps_amp_min_rows = 5
        model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(dtype="float32")]))

        def __init__(self):
            self.calls = 0
            self.fail_after = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def tok(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

        def predict(self, state, questions, **kwargs):
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError("simulated inference failure")
            return {
                "answers": {
                    "department": {
                        "type": "choice",
                        "choice": "billing",
                        "probabilities": {
                            "billing": 1.0,
                            "technical": 0.0,
                            "sales": 0.0,
                            "other": 0.0,
                        },
                    },
                    "churn_risk": {"type": "noul", "noul": 0.1},
                    "urgency": {
                        "type": "score",
                        "score": 0.3,
                        "probabilities": {"0": 0.8, "1": 0.1, "2": 0.1},
                    },
                },
                "usage": {"input_tokens": 30, "output_tokens": 0},
            }

    agent = Agent()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("USE_TF", "0")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    cli.prepare_runtime()
    monkeypatch.setattr("laya.load", lambda *args, **kwargs: agent)
    monkeypatch.setattr(cli, "download_checkpoint", lambda config: snapshot)
    monkeypatch.setattr(cli, "hardware", lambda: {"test": True})
    monkeypatch.setattr(cli, "command_output", lambda *args: None)
    args = argparse.Namespace(
        config=config_path,
        device="cpu",
        limit=2,
        instruction_language="both",
        output=tmp_path / "run",
    )
    return args, config, schemas, examples, agent


def test_complete_run_preserves_unicode_and_provenance(run_context, monkeypatch):
    args, config, schemas, examples, agent = run_context
    monkeypatch.setenv("HF_TOKEN", "test-only-auth-value")
    cli.evaluate(args, config, schemas, examples)
    metadata = json.loads((args.output / "metadata.json").read_text(encoding="utf-8"))
    predictions = (args.output / "predictions.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in predictions.splitlines()]
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))

    assert metadata["status"] == "complete"
    assert metadata["sample_count"] == 2
    assert metadata["group_count"] == 1
    assert metadata["decision_count"] == len(rows) == 12
    assert metadata["git_commit"] is None
    assert metadata["git_dirty"] is None  # Unavailable Git metadata is not a clean checkout.
    assert metadata["sha256"]["dataset"] == cli.fingerprint(Path(config["dataset"]))
    assert examples[0]["state"] in predictions
    assert "test-only-auth-value" not in json.dumps(metadata) + predictions + json.dumps(summary)
    assert {row["instruction_language"] for row in rows} == {"bn", "en"}
    assert all(summary[lang]["all"]["department"]["n"] == 2 for lang in ("bn", "en"))
    assert all(metadata["latency_ms"][lang]["requests"] == 2 for lang in ("bn", "en"))
    assert agent.calls == 6  # Four measured calls plus one warm-up per language.


def test_inference_failure_keeps_partial_run_and_marks_manifest(run_context):
    args, config, schemas, examples, agent = run_context
    agent.fail_after = 2  # Warm-up and first measured state succeed.
    with pytest.raises(RuntimeError, match="simulated inference failure"):
        cli.evaluate(args, config, schemas, examples)
    metadata = json.loads((args.output / "metadata.json").read_text(encoding="utf-8"))
    rows = (args.output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert metadata["status"] == "failed"
    assert "simulated inference failure" in metadata["error"]
    assert len(rows) == 3
    assert not (args.output / "summary.json").exists()


def test_existing_run_is_not_overwritten(run_context):
    args, config, schemas, examples, agent = run_context
    args.output.mkdir()
    sentinel = args.output / "metadata.json"
    sentinel.write_text('{"status":"complete"}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        cli.evaluate(args, config, schemas, examples)
    assert sentinel.read_text(encoding="utf-8") == '{"status":"complete"}'
    assert agent.calls == 0


def test_cli_config_error_has_no_traceback_or_inference(tmp_path, monkeypatch, capsys):
    path = tmp_path / "invalid.toml"
    path.write_text("revision = 123\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr("sys.argv", ["laya-bangla", "validate", "--config", str(path)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "revision" in error
    assert "Traceback" not in error
