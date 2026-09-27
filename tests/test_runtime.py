from pathlib import Path

import pytest

from laya_bangla.cli import CHECKPOINT_FILES, download_checkpoint


@pytest.mark.parametrize("missing_encoder", [False, True])
def test_download_pins_revision_and_requires_local_encoder(tmp_path, monkeypatch, missing_encoder):
    revision = "a" * 40
    snapshot = tmp_path / revision
    for name in CHECKPOINT_FILES:
        if missing_encoder and name == "encoder/config.json":
            continue
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    config = {"model_id": "example/checkpoint", "revision": revision}
    if missing_encoder:
        with pytest.raises(FileNotFoundError, match="encoder/config.json"):
            download_checkpoint(config)
    else:
        assert download_checkpoint(config) == Path(snapshot)
    assert calls[0]["revision"] == revision
    assert calls[0]["repo_id"] == "example/checkpoint"
    assert "encoder/config.json" in calls[0]["allow_patterns"]
