# Fine-tuning and benchmarking

This is the reproducible path from public data to a calibrated Bangla checkpoint
and its benchmark report. Each step writes to an ignored local directory and
records hashes of its inputs.

```bash
uv sync --locked --all-extras
uv run --frozen laya-bangla prepare                                   # data/processed/
uv run --frozen laya-bangla train --output models/laya-bangla-v1      # ~2 h on an RTX 4060
uv run --frozen laya-bangla benchmark --model base --output outputs/bench-base
uv run --frozen laya-bangla benchmark --model models/laya-bangla-v1 --output outputs/bench-v1
uv run --frozen laya-bangla compare outputs/bench-base/summary.json outputs/bench-v1/summary.json
uv run --frozen laya-bangla serve --model models/laya-bangla-v1       # http://127.0.0.1:8000
```

`prepare` needs network access once; later runs use `data/cache/`. Set
`HF_TOKEN` if anonymous Hugging Face downloads are rate-limited.

## Hardware

On Windows, `pyproject.toml` installs PyTorch from the CUDA 12.8 wheel index;
Linux wheels from PyPI already include CUDA, and macOS wheels include MPS. The
`train` command picks CUDA, then MPS, then CPU unless `--device` is given.

The released run used an NVIDIA GeForce RTX 4060 with 8 GB
of memory, of which training used at most the peak recorded in
`train_report.json`.

## Training method

`laya-bangla train` reuses upstream `laya.train` (Laya 0.4.11): its item
encoding, soft cross-entropy objective, temperature fitting, abstention
thresholds, and checkpoint format, so the result loads with `laya.load(path)`.
It adds what this dataset needs:

- **Fixed validation split.** Upstream `finetune` calibrates on a random slice
  of the training file. Here, model selection and calibration both use the
  prepared validation split, so test data never influences the checkpoint.
- **Best-epoch selection.** Validation accuracy and negative log-likelihood are
  measured before training and after every epoch. The epoch with the lowest
  validation NLL is restored before calibration.
- **Frozen token embeddings.** The 256k-token embedding matrix is two thirds of
  mmBERT-base's 307M parameters. Freezing it trains the other 125M (all
  transformer layers and the decision head), preserves the tokenizer's coverage
  of every language, and keeps peak memory well within 8 GB.
- **Option-order shuffling.** Choice options are presented in a new random
  order each epoch, so the model cannot learn the position prior upstream
  documents as an order-flip failure.

Hyperparameters live in [`configs/train.toml`](../configs/train.toml). The
output directory holds the checkpoint, `train_report.json` (data and config
hashes, validation history, selected epoch, calibration diagnostics, peak GPU
memory) and a copy of `tasks.json`.

## Benchmark protocol

`laya-bangla benchmark` scores a checkpoint, or the pinned upstream `base`, on:

- the held-out **test** split of every training source, with official test sets
  wherever the source has one;
- three **benchmark-only** sources never used for training (SentNoB,
  IndicSentiment, IndicXNLI);
- **end-to-end MASSIVE routing**, where the intent question uses the model's
  own predicted scenario;
- the 36-message **synthetic smoke set**, reported separately.

Every test question uses wording that never appears in training, in Bangla and
again in English. The summary reports accuracy, macro-F1, Brier score, NLL,
ECE, and ordinal MAE per source, task, wording language, and script slice
(`bangla`, `banglish`, `mixed`). Metric definitions are in
[local development](local-development.md#outputs-and-measurement-definitions).

Both checkpoints must be benchmarked on the same device: CUDA autocast can move
probabilities in the third decimal place relative to CPU.
