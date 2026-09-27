# laya-bangla

**A Bangla-focused adaptation of [Laya](https://huggingface.co/convaiinnovations/laya) for typed decisions, classification, and routing.**

বাংলা ভাষার জন্য Laya-কে আরো সুন্দর এবং সঠিক করার ছোট উদ্যক যেন ক্লাসিফিকেসন এ বাংলা ভাষা পিছিয়ে না থাকে।

[Roadmap](docs/roadmap.md) · [Contributing](CONTRIBUTING.md) · [Hugging Face model card template](docs/model-card-template.md) · [Apache 2.0 license](LICENSE)

## Overview

`laya-bangla` aims to improve Laya's performance on real-world Bangla (Bengali)
text through targeted data preparation, fine-tuning, and evaluation. The goal is
to make its decisions more useful for Bangla workflows and publish the resulting
fine-tuned checkpoint on Hugging Face.

Laya is a non-autoregressive decision model: it takes an input **state** and
**typed questions**, then returns structured answers and probabilities. Its
question types include `choice` (classification), `noul` (yes/no probability),
and `score` (ordinal scoring).

## Project status

**Local baseline development.** A pinned inference/evaluation toolchain and a
small Bangla diagnostic set are available. Fine-tuning and a released
`laya-bangla` checkpoint are still future milestones.

| Item | Status |
| --- | --- |
| Project documentation and repository conventions | Available |
| Local baseline CLI and locked Python environment | Available |
| Synthetic Bangla/Banglish/mixed-language smoke set | Available: 36 messages, 12 scenario groups |
| Reviewed training and held-out evaluation datasets | Planned |
| Fine-tuning and probability calibration | Planned |
| First local baseline measurements | [Completed on Apple M2 Pro / MPS](docs/baseline-smoke-results.md) |
| Fine-tuned Hugging Face model | Planned; repository ID to be selected |

Implementation milestones and open decisions are recorded in the
[roadmap](docs/roadmap.md).

## Why a Bangla adaptation?

Multilingual coverage alone does not establish accuracy on a particular language
or workflow. This project is motivated by weak Bangla results in the intended
use cases and will measure progress against a reproducible upstream baseline.

The intended coverage includes:

- **Bangla script:** everyday and formal writing, spelling variations, and
  colloquial expressions.
- **Banglish:** romanized Bangla with varied transliterations.
- **Code-switching:** Bangla and English mixed within the same input.
- **Local context:** names, places, Bangla numerals, dates, and domain vocabulary.
- **Typed decisions:** intent classification, request routing, yes/no decisions,
  and ordinal scoring.

The first diagnostic workflow is support-message routing, explicit threats to
leave a service, and urgency scoring. The [smoke-set notes](examples/README.md)
describe its scope. Representative datasets and release success criteria remain
to be chosen.

## Upstream model and planned starting point

The [Laya model family](https://huggingface.co/convaiinnovations/laya) is maintained
by Convai Innovations. Its model cards distinguish these checkpoints:

| Checkpoint | Upstream description | Role in this project |
| --- | --- | --- |
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | English-focused root checkpoint, ModernBERT-large | Original project reference |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | Multilingual checkpoint, mmBERT-base; includes Bangla | Pinned upstream baseline and planned fine-tuning starting point |

The multilingual checkpoint is also bundled under the `multilingual` subfolder
of the upstream family repository. The baseline uses immutable revision
`e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`, recorded in
[`configs/baseline.toml`](configs/baseline.toml). Each run records its source,
data, dependency-lock, and checkpoint file hashes.

## Getting started

### Explore this repository

```bash
git clone https://github.com/shamspias/laya-bangla.git
cd laya-bangla
```

Start with the [roadmap](docs/roadmap.md) for the development plan and
[contribution guide](CONTRIBUTING.md) for reporting Bangla failure cases.

### Run the local baseline

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to create the
isolated Python 3.12 environment from the committed dependency lock:

```bash
uv sync --locked
uv run --frozen laya-bangla doctor
uv run --frozen laya-bangla validate
uv run --frozen laya-bangla baseline --output outputs/baseline-001
```

The runner chooses Apple MPS, CUDA, or CPU based on availability. Pass
`--device mps` or `--device cpu` to select explicitly. The first run downloads
about 678 MB of upstream checkpoint/tokenizer files into the local ignored cache.
Choose a fresh output directory for every experiment.

Both Bangla and English question schemas are evaluated on the same 36 messages,
producing 216 typed decisions. Results include per-script/task metrics, raw
answers, timings, and a reproducibility manifest. This synthetic set is for
diagnostics, not a representative Bangla benchmark.

See [local development](docs/local-development.md) for command options,
measurement definitions, optional Hugging Face login, and download troubleshooting.

The [first local results](docs/baseline-smoke-results.md) show strong sensitivity
to question wording and poor churn/urgency decisions on several diagnostic
cases. They establish a starting point for data work and fine-tuning.

## Development checks

```bash
uv run --frozen laya-bangla validate
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

GitHub Actions is configured to run these checks and build the package on pushes
and pull requests. See [local development](docs/local-development.md) for clean
installation and model-backed verification commands.

## Contributing

Useful early contributions include Bangla failure cases, dataset suggestions,
annotation guidance, and evaluation ideas. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening an
[issue](https://github.com/shamspias/laya-bangla/issues) or pull request.

## License and acknowledgments

Original code and documentation in this repository are licensed under
[Apache 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.

This is an independent adaptation project based on Laya by **Convai Innovations**.
The upstream Laya model cards and source repository declare Apache 2.0 licensing.
Datasets and other third-party materials retain their own licenses; the eventual
model card will document the released checkpoint's provenance and license.

- [Upstream Laya source](https://github.com/NandhaKishorM/laya)
- [Laya model family](https://huggingface.co/convaiinnovations/laya)
- [Multilingual checkpoint](https://huggingface.co/convaiinnovations/laya-multilingual)
- [Upstream fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
