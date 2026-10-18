import math
from pathlib import Path

from laya_bangla.finetune import load_train_config, validation_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_validation_metrics_uses_each_items_own_options():
    records = [
        (0, [2.0, 0.0, 9.9], [1.0, 0.0], 2),  # logit beyond k is padding and ignored
        (1, [0.0, 1.0], [1.0, 0.0], 2),
    ]
    metrics = validation_metrics(records)
    assert metrics["accuracy"] == 0.5
    expected = (math.log(1 + math.exp(-2)) + math.log(1 + math.exp(1))) / 2
    assert math.isclose(metrics["nll"], expected)


def test_train_config_resolves_paths_and_is_valid_for_laya():
    from laya.train import TrainConfig

    config = load_train_config(ROOT / "configs/train.toml")
    assert config["baseline"] == (ROOT / "configs/baseline.toml").resolve()
    assert config["train"].name == "train.jsonl"
    trainer = TrainConfig(**dict(config["trainer"], seed=config["seed"]))
    trainer.shuffle_options = tuple(trainer.shuffle_options)
    trainer.validate()
    assert trainer.head_max_len < trainer.max_len
