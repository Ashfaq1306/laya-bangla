"""Hub publishing with a fake API: what is staged, verified, and tagged."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from laya_bangla import publish as p
from laya_bangla.prepare import sha256_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release(tmp_path):
    model = tmp_path / "model"
    for name in (*p.REQUIRED, "tokenizer/tokenizer_config.json"):
        (model / name).parent.mkdir(parents=True, exist_ok=True)
        (model / name).write_text(name, encoding="utf-8")
    (model / "optimizer.pt").write_text("not published", encoding="utf-8")
    run = tmp_path / "bench"
    run.mkdir()
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (run / "predictions.jsonl").write_text('{"state": "dataset text"}\n', encoding="utf-8")
    (run / "metadata.json").write_text("{}", encoding="utf-8")
    card = tmp_path / "card.md"
    card.write_text("---\nlicense: apache-2.0\n---\n\n# Card\n", encoding="utf-8")
    return model, card, {"base": run}, [ROOT / "LICENSE", ROOT / "NOTICE"]


def test_stage_publishes_checkpoint_card_and_metrics_only(release, tmp_path):
    model, card, benchmarks, notices = release
    staging = tmp_path / "staging"
    files = p.stage_model(model, card, benchmarks, notices, staging)
    assert files == sorted(
        [
            "LICENSE",
            "NOTICE",
            "README.md",
            "benchmark/base/summary.json",
            "encoder/config.json",
            "model.safetensors",
            "rl_agent_config.json",
            "tasks.json",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "train_report.json",
        ]
    )
    assert (staging / "README.md").read_text(encoding="utf-8") == card.read_text(encoding="utf-8")
    assert not list(staging.rglob("predictions.jsonl"))
    assert not (staging / "optimizer.pt").exists()


def test_stage_rejects_incomplete_inputs(release, tmp_path):
    model, card, benchmarks, notices = release
    with pytest.raises(ValueError, match="benchmark name"):
        p.stage_model(model, card, {"../x": benchmarks["base"]}, notices, tmp_path / "a")
    card.write_text("# No front matter\n", encoding="utf-8")
    with pytest.raises(ValueError, match="front-matter"):
        p.stage_model(model, card, benchmarks, notices, tmp_path / "b")
    (model / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        p.stage_model(model, card, benchmarks, notices, tmp_path / "c")


def test_published_card_has_front_matter_and_links_the_space():
    text = (ROOT / "docs/model-card.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "spaces/algolyzer/laya-bangla-playground" in text
    assert 'laya.load("algolyzer/laya-bangla", revision="v2")' in text


class FakeApi:
    def __init__(self, tags=(), corrupt=None):
        self.tags, self.corrupt, self.calls = list(tags), corrupt, []
        self.uploaded = {}

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def list_repo_refs(self, repo_id):
        return SimpleNamespace(tags=[SimpleNamespace(name=tag) for tag in self.tags])

    def upload_folder(self, repo_id, folder_path, **kwargs):
        self.calls.append(("upload_folder", repo_id, kwargs))
        for path in Path(folder_path).rglob("*"):
            if path.is_file():
                self.uploaded[path.relative_to(folder_path).as_posix()] = sha256_file(path)
        return SimpleNamespace(oid="abc123")

    def model_info(self, repo_id, revision, files_metadata):
        assert revision == "abc123" and files_metadata

        def sibling(name, digest):
            if name == self.corrupt:
                digest = "0" * 64
            lfs = SimpleNamespace(sha256=digest) if name.endswith(".safetensors") else None
            return SimpleNamespace(rfilename=name, lfs=lfs)

        return SimpleNamespace(siblings=[sibling(n, d) for n, d in self.uploaded.items()])

    def create_tag(self, repo_id, tag, revision):
        self.calls.append(("create_tag", tag, revision))


def publish(release, **kwargs):
    model, card, benchmarks, notices = release
    return p.publish_model(model, "org/model", card, benchmarks, notices, **kwargs)


def test_publish_model_uploads_verifies_and_tags(release):
    api = FakeApi()
    commit = publish(release, tag="v2", message="Upload", api=api)
    assert commit == "abc123"
    assert [call[0] for call in api.calls] == ["create_repo", "upload_folder", "create_tag"]
    assert api.calls[-1] == ("create_tag", "v2", "abc123")


def test_publish_model_never_moves_an_existing_tag(release):
    api = FakeApi(tags=["v2"])
    with pytest.raises(ValueError, match="already has a tag"):
        publish(release, tag="v2", message="Upload", api=api)
    assert "upload_folder" not in [call[0] for call in api.calls]


def test_publish_model_rejects_a_corrupted_upload(release):
    api = FakeApi(corrupt="model.safetensors")
    with pytest.raises(RuntimeError, match="does not match"):
        publish(release, tag="v2", message="Upload", api=api)
    assert "create_tag" not in [call[0] for call in api.calls]


def test_publish_space_uploads_the_gradio_app():
    api = FakeApi()
    assert p.publish_space(ROOT / "space", "org/space", "Update", api=api) == "abc123"
    assert api.calls[0] == (
        "create_repo",
        "org/space",
        {"repo_type": "space", "space_sdk": "gradio", "exist_ok": True},
    )
    assert {"app.py", "playground.py", "README.md", "requirements.txt"} <= set(api.uploaded)
    assert json.loads((ROOT / "space/support_questions.json").read_text(encoding="utf-8"))
