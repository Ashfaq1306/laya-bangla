# laya-bangla v2 model card (draft for Hugging Face)

This is the card for the first fine-tuned checkpoint, ready to become the model
repository's `README.md`. Before uploading: choose the repository ID, replace
`<repo-id>` and `<revision>`, and put the YAML block at the top without the
surrounding code fence. The checkpoint has not been published yet.

```yaml
---
language:
  - bn
  - en
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
base_model_relation: finetune
library_name: laya
pipeline_tag: text-classification
tags:
  - laya
  - bangla
  - bengali
  - banglish
  - typed-decisions
  - classification
  - routing
  - moderation
datasets:
  - mteb/amazon_massive_intent
  - mteb/sib200
  - aplycaebous/BanTH
  - aplycaebous/BanHate
  - TanjimKIT/Bangla_Sentiments_in_eLearning
---
```

## Model details

- **Name:** laya-bangla v2
- **Repository and revision:** `<repo-id>` @ `<revision>`
- **Source code:** https://github.com/shamspias/laya-bangla
- **Base checkpoint:** [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
  @ `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` (mmBERT-base encoder with Laya's
  typed decision head, 322M parameters)
- **Question types:** `choice`, `noul` (yes/no probability), and `score`
  (ordinal), unchanged from Laya.
- **Evaluated inputs:** Bangla script, romanized Bangla (Banglish), and
  Bangla-English mixed text; questions worded in Bangla or English.

laya-bangla adapts Laya, a non-autoregressive typed-decision model by Convai
Innovations, to Bangla routing, moderation, sentiment, and complaint triage.
The output format and loader are unchanged: `laya.load(path)` returns the same
`Agent` API as the base checkpoint.

## Intended use

- Routing Bangla or Banglish requests among a fixed set of options
  (assistant intents, complaint categories, departments).
- Moderation signals for Bangla and Banglish comments: hate speech, personal
  attacks, abusive or violent language.
- Three-level sentiment and severity scoring.

Calibrated probabilities make thresholds meaningful on the evaluated tasks. The
checkpoint also stores abstention thresholds fitted for a 10% target error.

## Limitations

- **Yes/no questions outside the trained tasks degrade.** On Bangla NLI
  (IndicXNLI), which it never trained on, accuracy falls from 0.69 to 0.53 with
  English wording because the model answers "no" too readily. Validate new
  `noul` questions before relying on them.
- **Rare positives are under-detected:** BanTH `abusive` macro-F1 is 0.51–0.59
  despite 0.83–0.92 accuracy.
- **Grievance severity** remains weak (0.40–0.43 accuracy over three levels).
- **Sentiment outside eLearning comments** transfers only partially (0.48–0.66).
- **No real customer-support data** was available; support-workflow behavior is
  untested beyond a 36-message synthetic smoke set.
- Code-mixed text is scarce in every training source (405 training rows).
- One training seed; no confidence intervals.

## Training data

Six published datasets, prepared by `laya-bangla prepare` with pinned revisions,
deduplication, removal of conflicting labels, group-aware splits, and removal of
any training text that appears in a validation or test split:

| Source | License | Typed decisions | Training rows |
| --- | --- | --- | ---: |
| MASSIVE `bn-BD` (Amazon) | CC BY 4.0 | assistant scenario (18), intent within scenario | 10,954 |
| SIB-200 `ben_Beng` | CC BY-SA 4.0 | topic (7) | 701 |
| BanglaPSG (Mendeley, DOI 10.17632/kcr8zfmtvb.1) | CC BY 4.0 | grievance category (6), severity (3 levels) | 4,156 |
| BanTH (transliterated Bangla) | MIT | hate, personal attack, abusive | 9,000 |
| BanHate | MIT | hate | 9,000 |
| Bangla Sentiments in eLearning (Bangla and romanized) | Apache-2.0 | sentiment (3 levels) | 4,955 |

That is 38,766 training rows (70,884 typed decisions) and a 2,597-row validation
split used only for epoch selection and calibration. Questions are asked with
two Bangla and one English wording; test wording is never seen in training.
Full details: [datasets](https://github.com/shamspias/laya-bangla/blob/main/docs/datasets.md).

## Training procedure

- **Code:** `laya-bangla train` with `configs/train.toml`, built on `laya.train`
  from Laya 0.4.11.
- **Method:** full fine-tuning of the transformer layers and decision head
  (125.3M trainable parameters) with the 256k-token embedding frozen. Soft
  cross-entropy with label smoothing 0.05; choice options shuffled each epoch.
- **Optimization:** AdamW, learning rate 3e-5 for the encoder and 1e-4 for the
  head, cosine decay to 1e-6, weight decay 0.01, effective batch 64 (32 × 2),
  2 epochs, fp16 autocast with gradient checkpointing, seed 20261018.
- **Budgets:** 512 tokens per sequence, 256 for questions and options.
- **Selection and calibration:** validation NLL after each epoch (epoch 2
  selected); per-question-type temperatures fitted on validation: choice 0.910,
  score 1.148, noul 1.178.
- **Hardware:** one NVIDIA GeForce RTX 4060 (8 GB); 53 minutes; peak 4.6 GB.

## Evaluation

Held-out test splits with unseen Bangla and English question wording, against
the base checkpoint on the same GPU. Each pair is Bangla wording / English
wording:

| Benchmark | n | Base accuracy | v2 accuracy | Base ECE | v2 ECE |
| --- | ---: | --- | --- | --- | --- |
| MASSIVE intent, end-to-end routing | 2,939 | 0.183 / 0.177 | **0.809 / 0.804** | — | — |
| MASSIVE scenario | 2,939 | 0.274 / 0.307 | **0.875 / 0.868** | 0.398 / 0.356 | 0.027 / 0.028 |
| BanglaPSG grievance category | 487 | 0.645 / 0.569 | **0.934 / 0.930** | 0.122 / 0.156 | 0.021 / 0.015 |
| BanglaPSG grievance severity | 487 | 0.339 / 0.398 | 0.433 / 0.398 | 0.302 / 0.456 | 0.031 / 0.038 |
| BanTH hate (Banglish) | 3,729 | 0.490 / 0.663 | **0.779 / 0.769** | 0.367 / 0.219 | 0.044 / 0.047 |
| BanHate hate | 2,720 | 0.610 / 0.555 | **0.736 / 0.740** | 0.202 / 0.316 | 0.061 / 0.091 |
| eLearning sentiment | 602 | 0.341 / 0.473 | **0.824 / 0.826** | 0.333 / 0.169 | 0.006 / 0.027 |
| SIB-200 topic | 204 | 0.574 / 0.578 | **0.765 / 0.775** | 0.283 / 0.285 | 0.076 / 0.094 |
| SentNoB sentiment (never trained) | 1,564 | 0.238 / 0.446 | 0.483 / 0.494 | 0.411 / 0.226 | 0.248 / 0.261 |
| IndicSentiment (never trained) | 996 | 0.012 / 0.466 | 0.594 / 0.663 | 0.684 / 0.237 | 0.121 / 0.075 |
| IndicXNLI entailment (never trained) | 3,340 | 0.596 / 0.694 | 0.499 / 0.529 | 0.190 / 0.134 | 0.256 / 0.209 |

Pooled over the six training sources' test splits (42,576 decisions), accuracy
rises from 53.0% to 82.0% and mean ECE falls from 0.273 to 0.037. The full
breakdown by task, wording, and script, with macro-F1 and ordinal MAE, is in
[benchmark results](https://github.com/shamspias/laya-bangla/blob/main/docs/benchmark-results.md).

## Usage

```bash
pip install "laya==0.4.11"
```

```python
import laya

agent = laya.load("<repo-id>", revision="<revision>")
questions = {
    "category": {
        "type": "choice",
        "instructions": "নাগরিকের এই সমস্যাটি কোন সেবা খাতের?",
        "criteria": {
            "water": "পানি সরবরাহ বা জলাবদ্ধতা",
            "electricity": "বিদ্যুৎ বা লোডশেডিং",
            "waste": "ময়লা, আবর্জনা বা পরিচ্ছন্নতা",
        },
    },
    "abusive": {
        "type": "noul",
        "instructions": "লেখাটি কি অশ্লীল গালি দেয় বা মারধরের কথা বলে?",
        "criteria": {"false": "না", "true": "হ্যাঁ"},
    },
}
result = agent.predict("আমাদের এলাকায় তিন দিন ধরে পানি নেই", questions)
print(result["answers"]["category"]["choice"], result["answers"]["abusive"]["noul"])
```

The checkpoint uses `max_len` 512 and `head_max_len` 256. It runs on CPU, CUDA,
or MPS; about 1.3 GB in float32.

## Artifacts

`model.safetensors` (fp16), `encoder/`, `tokenizer/`, `rl_agent_config.json`
(including fitted temperatures and abstention thresholds), `tasks.json` (the
question schemas used in training and evaluation), and `train_report.json`
(data and config hashes, validation history, calibration report).

## License and attribution

Released under Apache 2.0, the license of the base checkpoint and the Laya
source. Laya and `laya-multilingual` are by
[Convai Innovations](https://huggingface.co/convaiinnovations/laya)
([source](https://github.com/NandhaKishorM/laya)). Training data is credited
above under its own licenses: MASSIVE (FitzGerald et al., 2022), SIB-200
(Adelani et al., 2024), BanglaPSG, BanTH (NAACL 2025), BanHate,
and Bangla Sentiments in eLearning (Rahman et al., IEEE Access 2024). No
dataset text is distributed with the checkpoint.
