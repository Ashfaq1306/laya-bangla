# Local development and baseline evaluation

The first development environment is an **Apple M2 Pro, 16 GB RAM, macOS,
Python 3.12.12**. PyTorch detects its MPS GPU. The CLI also exposes CPU and CUDA
device selection; the dependency lock covers the pinned runtime.

The [first completed MPS run](baseline-smoke-results.md) records 216 decisions,
its configuration, timings, and observed failure cases.

## Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then run from the repository root:

```bash
uv sync --locked
uv run --frozen laya-bangla doctor
uv run --frozen laya-bangla validate
```

`uv` uses `.python-version`, installs into `.venv/`, and caches packages in
`.cache/uv/`. The CLI defaults `HF_HOME` to `.cache/huggingface/`, unless you
already set it. These local files are ignored by Git. No global Python packages
need to be changed. Development uses Python 3.12, even if `python3` on the machine
points to a newer interpreter.

## Run the checkpoint

```bash
# Full diagnostic pass: 36 states × 3 questions × 2 instruction languages.
uv run --frozen laya-bangla baseline --device mps --output outputs/baseline-mps-001

# A smaller CPU run, useful for checking the inference path on another machine.
uv run --frozen laya-bangla baseline --device cpu --limit 3 --output outputs/baseline-cpu-001
```

Omit `--device` for automatic CUDA → MPS → CPU selection. Explicitly requesting
an unavailable GPU fails with a message rather than labeling CPU work as a GPU
run. `--instruction-language bn` or `en` selects one question language; the
default is `both`. `--limit` is a debugging subset in file order, not sampling.

Use a **new output directory** each time. Existing directories are refused,
including directories left by interrupted runs. A `metadata.json` whose status
is still `running` is not evidence of a completed experiment.

### Model and dependency pins

[`../configs/baseline.toml`](../configs/baseline.toml) specifies:

- Model: `convaiinnovations/laya-multilingual`.
- Revision: `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`.
- Sequence budget: 1,024 tokens; question/option budget: 256 tokens.
- Seed: 42; CPU threads: 4; one unmeasured warm-up per instruction language.
- Ten equal-width calibration-error bins.

Runtime: Laya 0.3.20, PyTorch 2.8.0, Transformers 4.57.6, and Hugging Face Hub
0.36.0, with all resolved dependencies in `uv.lock`.

The PyPI Laya release does **not** accept `revision=` in `laya.load()`. The runner
uses `snapshot_download` with the immutable revision, checks for all required
checkpoint files, then loads that local directory. This also avoids downloading
an unpinned backbone/tokenizer as a fallback. Laya can normalize the local
tokenizer configuration for compatibility; the manifest hashes files after
that SDK normalization.

The first download is approximately 678 MB; model files are cached for later
runs. Model initialization and downloading are separate from measured request
latency. The CLI sets `USE_TF=0` before importing Laya to use its PyTorch runtime.

### Download troubleshooting

If the default Hugging Face Xet transfer stalls, retry with standard HTTP:

```bash
HF_HUB_DISABLE_XET=1 uv run --frozen laya-bangla baseline --device mps --output outputs/baseline-http-001
```

After an interrupted Xet download, force a clean HTTP download of the large
files first rather than reusing potentially incomplete transfer data:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_DISABLE_XET=1 uv run --frozen hf download \
  convaiinnovations/laya-multilingual \
  --revision e4e9ddf21a7b1903b7acffd8814ad4307bf63a67 \
  --include model.safetensors tokenizer/tokenizer.json --force-download
```

## Outputs and measurement definitions

- `metadata.json`: completion status, configuration, model revision, checkpoint
  and input hashes, code hashes, Git revision/dirty state, installed versions,
  hardware, precision settings, seeds, timings, and relevant runtime overrides.
- `predictions.jsonl`: UTF-8 inputs, scenario groups, question IDs, original SDK
  answers, expected/predicted labels, token usage, and per-request latency.
- `summary.json`: task metrics for each instruction language, both overall and
  separately for Bangla script, Banglish, and mixed inputs. Includes confusion
  matrices and majority-class reference accuracy.

Each request contains one state and all three questions, in the configured
order. GPU work is synchronized before and after timing. Latency includes
tokenization and answer decoding, excludes warm-up and loading, and is reported
per **three-question request**, not per individual decision. p95 uses nearest
rank. This is a small sequential diagnostic workload, not a throughput test.

Metrics use the SDK's four-decimal probabilities, renormalized to sum to one:

- **Accuracy / macro-F1:** reported `choice`, `noul >= 0.5`, and the highest
  probability ordinal level for `score`. F1 includes all declared classes,
  assigning zero when a class has no expected or predicted instances.
- **Score MAE:** absolute error of the SDK's expected ordinal score against the
  reference integer, averaged over score questions.
- **Brier score:** sum of squared errors over the entire class distribution,
  then mean over examples. Range 0–2, including binary questions; this is twice
  the single-probability binary convention.
- **NLL:** mean negative natural log probability of the reference label, with
  a `1e-12` floor for zero probabilities after SDK rounding.
- **ECE:** weighted difference between accuracy and mean maximum probability
  in ten equal-width bins. It uses maximum probability, **not** Laya's
  entropy-derived `confidence` field.

No temperature fitting is performed on this diagnostic set. The raw checkpoint
temperature configuration is retained in the manifest. Seeds and artifacts are
recorded for reproduction; GPU floating-point results can vary across hardware
and dependency versions.

## Checks

```bash
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

The tests cover label/split-group validation, calibration and ordinal metric
edge cases, and pinned checkpoint downloads without model inference.

## Next step toward fine-tuning

Use the failures from this smoke run to define annotation guidelines. Then
collect and review actual target-workflow data and construct group-aware
train/validation/calibration/test splits. The synthetic smoke set should remain
a diagnostic fixture. Choose the training method and local memory budget after
the reviewed data and a fixed held-out baseline are in place.
