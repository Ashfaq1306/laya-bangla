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

### Hugging Face login

The upstream baseline checkpoint is public, so authentication is optional.
For authenticated downloads and future model publishing, log in interactively
using the same cache location as the project:

```bash
HF_HOME="$PWD/.cache/huggingface" uv run --frozen hf auth login
HF_HOME="$PWD/.cache/huggingface" uv run --frozen hf auth whoami
```

Paste your token at the hidden login prompt. Token files stay under the ignored
`.cache/huggingface/` directory. An existing `HF_TOKEN` environment variable is
also supported by the Hub client; the runner does not record it in manifests.
If you choose another `HF_HOME`, use that value for both login and baseline runs.

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

Once the pinned checkpoint is cached, it can run completely offline:

```bash
HF_HUB_OFFLINE=1 uv run --frozen laya-bangla baseline --device mps --output outputs/baseline-offline-001
```

Offline mode fails if required files are missing instead of starting a download.

### Model and dependency pins

[`../configs/baseline.toml`](../configs/baseline.toml) specifies:

- Model: `convaiinnovations/laya-multilingual`.
- Revision: `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`.
- Sequence budget: 1,024 tokens; question/option budget: 256 tokens.
- Seed: 42; CPU threads: 4; one unmeasured warm-up per instruction language.
- Ten equal-width calibration-error bins.

Runtime: Laya 0.4.11, PyTorch 2.8.0, Transformers 4.57.6, and Hugging Face Hub
0.36.0, with all resolved dependencies in `uv.lock`. On Windows the lock resolves
PyTorch to the CUDA 12.8 build (`2.8.0+cu128`); PyPI's Windows wheels are CPU-only.

Laya was upgraded from 0.3.20 because 0.4.x ships the `laya.train` and
`laya.calibrate` modules the fine-tuning pipeline uses. Inference parity was
checked on 2026-10-18: the full 216-decision smoke run gives identical
accuracy, macro-F1, and ECE on CPU under 0.3.20 and 0.4.11, and matches the
recorded MPS run. On CUDA, fp16 autocast changes none of the 216 predicted
labels; it moves overall ECE by up to 0.023 and a per-slice ECE (12 decisions)
by up to 0.086, so compare checkpoints on one device.

The runner downloads the checkpoint itself rather than passing `revision=` to
`laya.load()` (which 0.3.20 did not support). The runner
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

The tests cover label/split-group validation, duplicate JSON keys, invalid
configuration types, calibration/ordinal metric edge cases, pinned downloads,
complete run artifacts, partial failures, and existing-output protection. They
use a simulated model for runner integration; real inference is a separate check.

The [GitHub Actions workflow](../.github/workflows/checks.yml) runs validation,
tests, lint, formatting, and distribution builds on pushes and pull requests.
Hub network access is disabled in these checks; model weights are unnecessary.

### Verify a clean, non-editable installation

To test the built package in a separate environment, with dependencies from the
same lockfile, run these commands from the checkout root:

```bash
UV_PROJECT_ENVIRONMENT=.cache/verify-venv uv sync --locked --no-editable
UV_PROJECT_ENVIRONMENT=.cache/verify-venv uv run --frozen --no-editable laya-bangla validate
UV_PROJECT_ENVIRONMENT=.cache/verify-venv uv run --frozen --no-editable pytest -q
uv build
```

This installs the project as a wheel rather than linking imports directly to
`src/`. Run the CLI from the checkout root so it can find the versioned configs
and example data, or pass an explicit `--config` path. Source and wheel
distribution artifacts are written to ignored `dist/`.

### Verified on the development machine

On 2026-09-27, the updated runner was installed as a non-editable wheel in
`.cache/verify-venv/` and checked on the Apple M2 Pro with Python 3.12.12:

| Check | Result |
| --- | --- |
| Locked environment and schema/data validation | Passed |
| Unit and runner-integration tests | 26 passed |
| Ruff lint and formatting | Passed |
| Source distribution and wheel build | Passed; license and attribution included |
| GitHub workflow syntax (`actionlint` 1.7.7) | Passed |
| Offline CPU inference | 3 states, both schemas, 18 decisions completed |
| Offline MPS inference | 36 states, both schemas, 216 decisions completed |

The model-backed runs are saved locally under `outputs/verification-cpu-v1/`
and `outputs/verification-mps-v1/`. The full MPS `summary.json` exactly matches
the first diagnostic run's metric summary; request timings differ. These remain
synthetic diagnostics, not a new benchmark or fine-tuning result.

The GitHub workflow has been validated locally; its first hosted run occurs
after the workflow is pushed. CUDA inference has not been exercised on this
Apple development machine.

## Fine-tuning

Data preparation, training, benchmarking, and the web playground are described
in [fine-tuning and benchmarking](fine-tuning.md); the data sources are in
[datasets](datasets.md). The synthetic smoke set remains a diagnostic fixture
and is never trained on.
