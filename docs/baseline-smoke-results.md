# First local baseline: synthetic smoke set v1

**Completed:** 2026-09-27, 02:06 UTC, as recorded in the run manifest.

This run verifies that the pinned upstream multilingual checkpoint can be
evaluated on the local Apple GPU. It measures **36 synthetic diagnostic
messages from 12 scenario groups**, with provisional labels and no independent
speaker review. These are smoke-test observations, not a representative Bangla
benchmark or fine-tuned model results. See the [data notes](../examples/README.md).

## Reproduce

From the repository root, after `uv sync --locked`:

```bash
HF_HUB_DISABLE_XET=1 uv run --frozen laya-bangla baseline --device mps --output outputs/baseline-mps-v3
```

That output directory exists on the development machine; choose a new name for
another run. Local artifacts are `metadata.json`, `predictions.jsonl`, and
`summary.json` under `outputs/baseline-mps-v3/`. They are ignored by Git; this
curated report records the initial observations.

## Configuration

| Setting | Value |
| --- | --- |
| Model | `convaiinnovations/laya-multilingual` |
| Immutable revision | `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` |
| Hardware | Apple M2 Pro, 16 GB unified RAM, 12 logical CPU cores |
| Platform | macOS 26.6.2, arm64, Python 3.12.12 |
| Runtime | Laya 0.3.20, PyTorch 2.8.0, Transformers 4.57.6, Hub 0.36.0 |
| Additional numeric dependencies | NumPy 2.5.3, safetensors 0.8.0 |
| Device / precision | MPS; float32 model parameters and forward passes |
| Inputs | 12 Bangla-script, 12 Banglish, 12 mixed-language states |
| Questions | Department (`choice`), churn threat (`noul`), urgency (`score`) |
| Instructions | Same tasks with Bangla and English question/criterion wording |
| Request shape | One state and three questions, sequential requests |
| Token limits | 1,024 total; 256 for question/option text |
| Seed / CPU threads | 42 / 4 |
| Calibration | Upstream temperatures all 1.0; no new fitting |
| Total decisions | 36 × 3 × 2 = 216 |

Laya enables MPS autocast only at five or more question rows by default. These
requests had three rows, so the forward passes used float32 despite the stored
autocast dtype being float16. Model loading took 16.33 seconds with files already
downloaded; the complete process took approximately 32.9 seconds.

The run was made from an uncommitted baseline implementation on top of setup
commit `4df155e32f8f87c71136f179bcc1a78c20a3edfc`. The manifest records the dirty
state and SHA-256 hashes of each source/input file, including:

```text
dataset:       6dc93889deb5724e7feb98866bf069c9fbf214c4a854c0d260f22553c2e9d0e7
questions_bn:  9c7f104effefe67ef6f69b9923c5205249985de1b69f6cbef746910ee9697083
questions_en:  b9f16390263434ba01d4dbb0348cabb3b7cd6803defd13c7e688e2d72f5db50c
uv.lock:       03887d6a162bb3976ecc94b564761d820063c47e5406c4b6203f6afdd1ef75b2
source/cli.py: 2198d2fdd6f17ed7cfc1f117826495f4d3c7b775e86c7181897fa74c7a914b01
model weights: 9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204
```

## Overall results

Each row contains 36 decisions. Accuracy for urgency uses the most probable
ordinal level; MAE uses the SDK's expected score on the 0–2 scale.

| Question wording | Task | Correct | Accuracy | Macro-F1 | Brier | ECE | Score MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Bangla | Department | 25/36 | 69.4% | 0.653 | 0.399 | 0.142 | — |
| English | Department | 27/36 | 75.0% | 0.741 | 0.348 | 0.114 | — |
| Bangla | Churn threat | 18/36 | 50.0% | 0.333 | 0.903 | 0.479 | — |
| English | Churn threat | 20/36 | 55.6% | 0.518 | 0.818 | 0.422 | — |
| Bangla | Urgency | 15/36 | 41.7% | 0.352 | 0.701 | 0.172 | 0.741 |
| English | Urgency | 13/36 | 36.1% | 0.221 | 0.803 | 0.321 | 0.685 |

Majority-class reference accuracies are 25% for department, 50% for churn, and
33.3% for urgency. Brier is the multiclass sum convention (0–2 even for binary
questions). ECE uses ten equal-width bins and maximum answer probability.
Complete definitions are in [local development](local-development.md).

### Accuracy by input form

Each row contains 12 states; each task is evaluated once per state.

| Question wording | Input form | Department | Churn threat | Urgency |
| --- | --- | --- | --- | --- |
| Bangla | Bangla script | 10/12 | 6/12 | 8/12 |
| Bangla | Banglish | 8/12 | 6/12 | 3/12 |
| Bangla | Mixed | 7/12 | 6/12 | 4/12 |
| English | Bangla script | 9/12 | 6/12 | 4/12 |
| English | Banglish | 8/12 | 8/12 | 4/12 |
| English | Mixed | 10/12 | 6/12 | 5/12 |

### Observed request latency

| Question wording | Requests | Median | p95 |
| --- | --- | --- | --- |
| Bangla | 36 | 86.7 ms | 313.1 ms |
| English | 36 | 69.0 ms | 397.6 ms |

These are synchronized, three-question request timings including tokenization
and decoding, after one warm-up per instruction language. Bangla ran first and
sequence lengths varied; first-use work for new shapes can remain. This small
single run does not establish a stable throughput or language-speed comparison.

## Failure cases to investigate

1. **Churn answers are highly sensitive to question wording.** Bangla wording
   returned `true` for every state, including all 18 negative examples. English
   wording returned `false` on every Bangla-script and mixed-language state.
   High confidence did not imply correctness.
2. **Negation is a useful diagnostic target.** `technical-01-bn` explicitly says
   “সেবা ছেড়ে যাওয়ার ইচ্ছা নেই।” With Bangla questions, `P(churn)` was 0.7615.
   `billing-01-bn` contains no churn threat, yet received 0.9765.
3. **Ordinal predictions collapse toward a few levels.** English wording chose
   urgency level 1 in 35/36 cases and never chose level 0. Bangla wording chose
   level 0 only once, and Banglish urgency accuracy was below the majority
   reference on these cases.
4. **General/other routing needs better coverage.** With Bangla wording, only
   2/9 `other` cases were routed correctly, versus 9/9 technical cases. English
   wording improved `other` to 5/9 but still missed all three Bangla-script cases.

The paired examples are correlated and labels are provisional. No confidence
intervals or generalization claims are drawn from this small constructed set.
The differences above should guide more controlled experiments and annotation,
not prompt selection against a final test set.

## Next work

- Review the labels and question translations with fluent Bangla speakers.
- Collect representative target-workflow data, especially negative/conditional
  churn statements, low-urgency messages, varied transliterations, and `other`
  routing cases. Record provenance and dataset permissions.
- Establish group-aware train, validation/calibration, and untouched test splits.
- Diagnose `noul` wording/label effects and ordinal position effects on a separate
  development set while preserving this baseline for comparison.
- Measure local training memory with the chosen adaptation method once the
  reviewed data and a fixed held-out baseline are ready.
