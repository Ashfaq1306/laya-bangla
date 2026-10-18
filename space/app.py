"""Hugging Face Space: try laya-bangla in real time and compare it with the base model."""

import json
import os
import threading
import time
from pathlib import Path

import gradio as gr
import laya
from huggingface_hub import hf_hub_download
from playground import (
    BASE_REPO,
    BASE_REVISION,
    BENCHMARKS,
    CUSTOM_TEMPLATE,
    EXAMPLES,
    GITHUB,
    MODEL_REPO,
    MODEL_REVISION,
    PRESET_LABELS,
    TRAINED_SOURCES,
    UNSEEN,
    benchmark_table,
    intent_question,
    load_support_questions,
    parse_questions,
    pooled_accuracy,
    preset_questions,
    result_card,
)

# A local checkpoint directory can stand in for the Hub repository when testing.
MODEL = os.environ.get("LAYA_BANGLA_MODEL", MODEL_REPO)
REVISION = os.environ.get("LAYA_BANGLA_REVISION", MODEL_REVISION)


def model_file(name: str) -> Path:
    if Path(MODEL).is_dir():
        return Path(MODEL) / name
    return Path(hf_hub_download(MODEL, name, revision=REVISION))


def read_json(name: str) -> dict:
    return json.loads(model_file(name).read_text(encoding="utf-8"))


TASKS = read_json("tasks.json")
SUPPORT = load_support_questions()
BASE_SUMMARY = read_json("benchmark/base/summary.json")
TUNED_SUMMARY = read_json("benchmark/laya-bangla-v2/summary.json")
AGENTS = {
    "laya-bangla": laya.load(MODEL, revision=None if Path(MODEL).is_dir() else REVISION),
    "base": laya.load(BASE_REPO, revision=BASE_REVISION),
}
SUBTITLES = {"laya-bangla": "fine-tuned for Bangla", "base": "laya-multilingual, before tuning"}
LOCK = threading.Lock()  # Laya agents are not documented as thread-safe.


def questions_for(preset: str, lang: str) -> str:
    if preset == "custom":
        return json.dumps(CUSTOM_TEMPLATE, ensure_ascii=False, indent=2)
    return json.dumps(preset_questions(TASKS, SUPPORT, preset, lang), ensure_ascii=False, indent=2)


def run(text: str, preset: str, lang: str, questions_json: str, compare: bool) -> str:
    text = (text or "").strip()
    if not text:
        return '<p class="lb-note">Type or pick a message to see the decisions.</p>'
    if len(text) > 2000:
        raise gr.Error("Please keep the text under 2,000 characters.")
    try:
        questions = parse_questions(questions_json)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    cards = []
    for name, agent in AGENTS.items():
        if name == "base" and not compare:
            continue
        start = time.perf_counter()
        try:
            with LOCK:
                answers = agent.predict(text, questions)["answers"]
                shown = dict(questions)
                if preset == "assistant" and "massive_scenario" in answers:
                    follow_up = intent_question(TASKS, answers["massive_scenario"]["choice"], lang)
                    if follow_up:
                        answers.update(agent.predict(text, follow_up)["answers"])
                        shown.update(follow_up)
        except (ValueError, TypeError, KeyError) as exc:
            raise gr.Error(f"The model could not answer these questions: {exc}") from exc
        ms = (time.perf_counter() - start) * 1000
        cards.append(result_card(name, SUBTITLES[name], answers, shown, ms))
    return f'<div class="lb-grid">{"".join(cards)}</div>'


INTRO = f"""
# laya-bangla · বাংলা সিদ্ধান্ত মডেল

Write a message in **Bangla, Banglish (Bangla in English letters), or a mix**. The model answers
fixed questions about it — which department should handle it, is it hate speech, how urgent is
it — and shows how sure it is about every option. The **base model** it was trained from is shown
next to it, so you can see what fine-tuning changed.

[Model card](https://huggingface.co/{MODEL_REPO}) · [Source code]({GITHUB}) ·
Base model: [{BASE_REPO}](https://huggingface.co/{BASE_REPO})
"""

HOW_TO = f"""
## Use the model in Python

```bash
pip install "laya==0.4.11"
```

```python
import laya

agent = laya.load("{MODEL_REPO}", revision="{MODEL_REVISION}")

questions = {{
    "sector": {{
        "type": "choice",
        "instructions": "নাগরিকের এই সমস্যাটি কোন সেবা খাতের?",
        "criteria": {{
            "water": "পানি সরবরাহ বা জলাবদ্ধতা",
            "electricity": "বিদ্যুৎ বা লোডশেডিং",
            "waste": "ময়লা, আবর্জনা বা পরিচ্ছন্নতা",
        }},
    }},
    "urgent": {{
        "type": "noul",
        "instructions": "এটি কি এখনই সমাধান করা দরকার?",
        "criteria": {{"false": "না", "true": "হ্যাঁ"}},
    }},
}}

result = agent.predict("আমাদের এলাকায় তিন দিন ধরে পানি নেই", questions)
print(result["answers"]["sector"]["choice"])   # water
print(result["answers"]["urgent"]["noul"])     # probability of "yes", 0 to 1
```

## The three question types

| Type | Answers | Example |
| --- | --- | --- |
| `choice` | one option from a list, with a probability for each | Which department? billing / technical / sales |
| `noul` | the probability that the answer is **yes** | Is this hate speech? |
| `score` | a level on an ordered scale, with a probability per level | How urgent? low / medium / high |

Write the `instructions` and option descriptions in Bangla or English; both work. Every question
in one call is answered from the same text. The **Write your own questions** option in the
*Try it* tab accepts the same JSON format.

## Tips

- Probabilities are calibrated on held-out data for the trained tasks: 80% means roughly
  8 correct out of 10. For a new kind of yes/no question, check a few answers first (see the
  *Data & limits* tab).
- Short option descriptions that clearly differ from each other work best.
- Runs on CPU, CUDA, or Apple MPS. This Space runs on a shared CPU, so answers take about a second
  per model.
"""

DATA_AND_LIMITS = f"""
## Datasets used for training

All training data is real, published Bangla text. Nothing is machine-generated.

| Dataset | What the text is | What the model learned | License |
| --- | --- | --- | --- |
| [MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) `bn-BD` | Voice-assistant requests | Scenario (18) and intent | CC BY 4.0 |
| [SIB-200](https://huggingface.co/datasets/mteb/sib200) `ben_Beng` | News-style sentences | Topic (7) | CC BY-SA 4.0 |
| [BanglaPSG](https://data.mendeley.com/datasets/kcr8zfmtvb) | Citizen complaints from social media | Service sector (6), severity (3 levels) | CC BY 4.0 |
| [BanTH](https://huggingface.co/datasets/aplycaebous/BanTH) | YouTube comments in **Banglish** | Hate, personal attack, abusive | MIT |
| [BanHate](https://huggingface.co/datasets/aplycaebous/BanHate) | YouTube comments in Bangla script | Hate | MIT |
| [Bangla Sentiments in eLearning](https://huggingface.co/datasets/TanjimKIT/Bangla_Sentiments_in_eLearning) | Course comments, Bangla and romanized | Sentiment (3 levels) | Apache-2.0 |

Three more datasets are used **only for testing**, never for training:
[SentNoB](https://huggingface.co/datasets/khondoker/SentNoB),
[IndicSentiment](https://huggingface.co/datasets/mteb/IndicSentiment), and
[IndicXNLI](https://huggingface.co/datasets/mteb/IndicXnliPairClassification).
The [dataset guide]({GITHUB}/blob/main/docs/datasets.md) explains every cleaning and split step.

## Known limits

- **New kinds of yes/no questions can get worse.** All trained yes/no tasks are about moderation,
  so the model learned to say "no" readily. On a sentence-logic test it never trained on
  (IndicXNLI) it drops to about 50–53%, below the base model. Test new yes/no questions first.
- **Rare cases are missed.** Abusive comments are only about 8% of BanTH; the model finds few of them.
- **Complaint severity is weak** (about 40–43% over three levels).
- **Voice-assistant routing was trained on Bangla script only.** Banglish commands such as
  "gaan chalao" are often routed to the wrong scenario.
- **Sentiment outside course comments** is only partly learned (48–66%).
- **No real customer-support data** was available. The *Customer support* preset is a
  diagnostic, not a measured result, and laya-bangla often misses threats to leave there.
- Mixed Bangla-English text is rare in the training data.

The [model card](https://huggingface.co/{MODEL_REPO}) has the full details.
"""


def benchmark_page() -> str:
    tuned_in = pooled_accuracy(TUNED_SUMMARY, TRAINED_SOURCES)
    base_in = pooled_accuracy(BASE_SUMMARY, TRAINED_SOURCES)
    return f"""
## How well does it work?

Every number here is measured on **test data the model never saw** during training, and every
question uses **wording it never saw** during training. Each question is asked twice: once
worded in Bangla and once in English. Cells show **accuracy with the Bangla question / with the
English question**.

**Overall: {base_in:.0%} → {tuned_in:.0%}** correct over all test decisions of the six
training datasets.

### Tasks it was trained on

{benchmark_table(BASE_SUMMARY, TUNED_SUMMARY, BENCHMARKS)}

### Datasets it never trained on

These check whether the model still works outside its training data.

{benchmark_table(BASE_SUMMARY, TUNED_SUMMARY, UNSEEN)}

The base model and laya-bangla were tested on the same GPU with the same files. The
"two steps" routing row first predicts the scenario and then the intent within it, as a real
assistant would. Full tables with macro-F1, calibration error, and results by script are in the
[benchmark report]({GITHUB}/blob/main/docs/benchmark-results.md).
"""


CSS = """
.lb-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.lb-card { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 12px; }
.lb-head { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
.lb-head span, .lb-qname, .lb-pct { color: var(--body-text-color-subdued); font-size: 0.85em; }
.lb-q { border-top: 1px solid var(--border-color-primary); padding: 8px 0; }
.lb-instr { font-size: 0.9em; margin: 2px 0 4px; }
.lb-answer { font-weight: 600; margin-bottom: 6px; }
.lb-row { display: grid; grid-template-columns: minmax(0, 1fr) 80px 48px; gap: 6px;
  align-items: center; font-size: 0.85em; }
.lb-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lb-bar { height: 8px; border-radius: 4px; background: var(--background-fill-secondary); }
.lb-bar i { display: block; height: 100%; border-radius: 4px; background: var(--color-accent); }
.lb-pct { text-align: right; }
"""

with gr.Blocks(title="laya-bangla playground") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("Try it · চেষ্টা করুন"):
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(
                    label="Your text (Bangla, Banglish, or mixed)",
                    placeholder="যেমন: বিলে দুইবার টাকা কেটেছে, আজই ফেরত চাই",
                    lines=3,
                    max_length=2000,
                )
                preset = gr.Dropdown(
                    [(label, key) for key, label in PRESET_LABELS.items()],
                    value="grievance",
                    label="What should the model decide?",
                )
                lang = gr.Radio(
                    [("বাংলা", "bn"), ("English", "en")],
                    value="bn",
                    label="Language of the questions",
                )
                with gr.Row():
                    live = gr.Checkbox(value=True, label="Answer as I type")
                    compare = gr.Checkbox(value=True, label="Also show the base model")
                button = gr.Button("Get decisions", variant="primary")
                with gr.Accordion("Questions sent to the model (editable JSON)", open=False):
                    questions = gr.Code(
                        questions_for("grievance", "bn"), language="json", interactive=True
                    )
            with gr.Column(scale=3):
                output = gr.HTML(run("", "grievance", "bn", "{}", True))
        examples = gr.Examples(
            [list(e) for e in EXAMPLES], inputs=[text, preset], label="Try an example"
        )
        gr.Markdown(
            "The built-in questions use the benchmark's test wording, which the model never saw "
            "in training. *Customer support* questions come from a small synthetic diagnostic set."
        )
    with gr.Tab("Benchmark results"):
        gr.Markdown(benchmark_page())
    with gr.Tab("How to use"):
        gr.Markdown(HOW_TO)
    with gr.Tab("Data & limits"):
        gr.Markdown(DATA_AND_LIMITS)

    inputs = [text, preset, lang, questions, compare]
    shared = {"concurrency_id": "predict", "concurrency_limit": 1}
    for trigger in (button.click, text.submit, compare.input):
        trigger(run, inputs, output, **shared)
    # While typing, only the newest text is answered once the model is free.
    text.input(
        lambda live, *args: run(*args) if live else gr.skip(),
        [live, *inputs],
        output,
        trigger_mode="always_last",
        show_progress="minimal",
        **shared,
    )
    for event in (preset.input, lang.input, examples.load_input_event.then):
        event(questions_for, [preset, lang], questions).then(run, inputs, output, **shared)

if __name__ == "__main__":
    demo.queue(max_size=32).launch(theme=gr.themes.Soft(), css=CSS)
