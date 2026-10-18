"""Questions, routing, and result formatting for the Hugging Face Space (no Gradio import).

This mirrors `laya_bangla.app`, which the Space cannot import because the package is not on
PyPI; `tests/test_space.py` checks that both ask the model exactly the same questions.
"""

import html
import json
from pathlib import Path

MODEL_REPO = "algolyzer/laya-bangla"
MODEL_REVISION = "v2"
BASE_REPO = "convaiinnovations/laya-multilingual"
BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
GITHUB = "https://github.com/shamspias/laya-bangla"

HERE = Path(__file__).parent
PRESETS = {
    "support": ("department", "churn_risk", "urgency"),
    "grievance": ("grievance_category", "grievance_severity"),
    "moderation": ("hate", "personal_attack", "abusive"),
    "sentiment": ("sentiment",),
    "assistant": ("massive_scenario",),
    "topic": ("sib_topic",),
}
PRESET_LABELS = {
    "support": "Customer support: department, threat to leave, urgency",
    "grievance": "Citizen complaint: service sector and severity",
    "moderation": "Moderation: hate speech, personal attack, abuse",
    "sentiment": "Sentiment: negative, neutral, or positive",
    "assistant": "Voice assistant: what is the user asking for?",
    "topic": "Topic of a sentence",
    "custom": "Write your own questions",
}
EXAMPLES = [
    ("আমাদের এলাকায় তিন দিন ধরে পানি নেই, বাচ্চারা অসুস্থ হয়ে পড়ছে।", "grievance"),
    ("রাস্তার পাশে ময়লার স্তূপ, দুর্গন্ধে হাঁটা যায় না।", "grievance"),
    ("কাল সকাল সাতটায় আমাকে জাগিয়ে দিও", "assistant"),
    ("আজকের আবহাওয়া কেমন? বৃষ্টি হবে?", "assistant"),
    ("Sir apnar class gulo onek helpful, onek dhonnobad!", "sentiment"),
    ("ডেলিভারি তিন দিন দেরি, খাবারও ঠান্ডা ছিল। খুবই হতাশ।", "sentiment"),
    ("Tor moto beyadob manush ke deshe rakha uchit na", "moderation"),
    ("বিশ্বকাপ ফাইনালে আর্জেন্টিনা টাইব্রেকারে জয় পেয়েছে।", "topic"),
    ("বিলে দুইবার টাকা কেটেছে, আজকের মধ্যে ফেরত না দিলে অন্য কোম্পানিতে চলে যাব।", "support"),
    ("App e login korte parchi na, kal theke kaj atke ache. Please ajkei fix korun.", "support"),
]
CUSTOM_TEMPLATE = {
    "is_question": {
        "type": "noul",
        "instructions": "লেখাটি কি কোনো প্রশ্ন?",
        "criteria": {"false": "না, এটি প্রশ্ন নয়", "true": "হ্যাঁ, এটি একটি প্রশ্ন"},
    },
    "product": {
        "type": "choice",
        "instructions": "লেখাটি কোন পণ্য বা সেবা নিয়ে?",
        "criteria": {
            "mobile": "মোবাইল ফোন বা সিম",
            "internet": "ইন্টারনেট বা ওয়াইফাই",
            "bank": "ব্যাংক বা মোবাইল ব্যাংকিং",
            "other": "অন্য কিছু",
        },
    },
}


def load_support_questions(path: Path = HERE / "support_questions.json") -> dict:
    """The diagnostic support schemas, copied from `configs/questions.{bn,en}.json`."""
    return json.loads(path.read_text(encoding="utf-8"))


def render_question(task: dict, instructions: str, lang: str, options) -> dict:
    if task["type"] == "score":
        criteria = [option[lang] for option in options]
    else:
        criteria = {label: option[lang] for label, option in options.items()}
    return {"type": task["type"], "instructions": instructions, "criteria": criteria}


def preset_questions(tasks: dict, support: dict, preset: str, lang: str) -> dict:
    """Questions for a preset, using the evaluation wording (never seen in training)."""
    if preset == "support":
        return support[lang]
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}")
    questions = {}
    for name in PRESETS[preset]:
        task = tasks[name]
        questions[name] = render_question(
            task, task["instructions"]["eval"][lang], lang, task["criteria"]
        )
    return questions


def intent_question(tasks: dict, scenario: str, lang: str) -> dict | None:
    """Second routing stage: the intent among the predicted MASSIVE scenario's options."""
    task = tasks["massive_intent"]
    options = task["groups"].get(scenario)
    if not options:
        return None
    return {
        "massive_intent": render_question(task, task["instructions"]["eval"][lang], lang, options)
    }


def parse_questions(text: str) -> dict:
    """Validate a user-written question schema before it reaches the model."""
    try:
        questions = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Questions are not valid JSON: {exc.msg} (line {exc.lineno})") from exc
    if not isinstance(questions, dict) or not questions:
        raise ValueError("Questions must be a JSON object with at least one question")
    if len(questions) > 8:
        raise ValueError("Ask at most 8 questions at once")
    for name, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(f"{name}: each question must be an object")
        kind = question.get("type")
        if kind not in ("choice", "noul", "score"):
            raise ValueError(f"{name}: type must be choice, noul, or score")
        if not str(question.get("instructions", "")).strip():
            raise ValueError(f"{name}: instructions are required")
        criteria = question.get("criteria")
        if kind == "score" and not (isinstance(criteria, list) and len(criteria) >= 2):
            raise ValueError(f"{name}: a score question needs a list of at least 2 levels")
        if kind != "score" and not (isinstance(criteria, dict) and len(criteria) >= 2):
            raise ValueError(f"{name}: criteria must map at least 2 option names to descriptions")
        if kind == "noul" and set(criteria) != {"false", "true"}:
            raise ValueError(f"{name}: a noul question needs exactly the options false and true")
    return questions


def answer_rows(answer: dict, question: dict) -> tuple[str, list[tuple[str, float]]]:
    """The predicted label and every option's probability, most likely first."""
    if answer["type"] == "noul":
        yes = float(answer["noul"])
        labels = question["criteria"]
        rows = [(f"true: {labels['true']}", yes), (f"false: {labels['false']}", 1 - yes)]
        return ("Yes / হ্যাঁ" if yes >= 0.5 else "No / না"), sorted(rows, key=lambda r: -r[1])
    probabilities = {key: float(value) for key, value in answer["probabilities"].items()}
    if answer["type"] == "score":
        levels = question["criteria"]
        rows = [(f"{i}: {levels[i]}", probabilities[str(i)]) for i in range(len(levels))]
        top = max(range(len(levels)), key=lambda i: probabilities[str(i)])
        return f"Level {top}: {levels[top]}", rows
    criteria = question["criteria"]
    rows = [(f"{key}: {criteria.get(key, key)}", value) for key, value in probabilities.items()]
    choice = answer["choice"]
    return f"{choice}: {criteria.get(choice, choice)}", sorted(rows, key=lambda r: -r[1])


def result_card(title: str, subtitle: str, answers: dict, questions: dict, ms: float) -> str:
    """One model's answers as HTML: the decision, then a probability bar per option."""
    parts = [
        f'<div class="lb-card"><div class="lb-head"><b>{html.escape(title)}</b>'
        f"<span>{html.escape(subtitle)} · {ms:.0f} ms</span></div>"
    ]
    for name, question in questions.items():
        if name not in answers:
            continue
        decision, rows = answer_rows(answers[name], question)
        parts.append(
            f'<div class="lb-q"><div class="lb-qname">{html.escape(name)} · '
            f"{html.escape(question['type'])}</div>"
            f'<div class="lb-instr">{html.escape(question["instructions"])}</div>'
            f'<div class="lb-answer">{html.escape(decision)}</div>'
        )
        for label, value in rows:
            parts.append(
                f'<div class="lb-row"><span class="lb-label">{html.escape(label)}</span>'
                f'<span class="lb-bar"><i style="width:{value * 100:.1f}%"></i></span>'
                f'<span class="lb-pct">{value * 100:.1f}%</span></div>'
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


# (summary key, plain-language task, dataset, script)
BENCHMARKS = [
    ("massive/intent (end-to-end)", "Voice-assistant routing, two steps", "MASSIVE", "Bangla"),
    ("massive/massive_scenario", "Voice-assistant scenario (18 options)", "MASSIVE", "Bangla"),
    ("banglapsg/grievance_category", "Complaint service sector (6)", "BanglaPSG", "Bangla"),
    ("banglapsg/grievance_severity", "Complaint severity (3 levels)", "BanglaPSG", "Bangla"),
    ("banth/hate", "Hate speech", "BanTH", "Banglish"),
    ("banth/personal_attack", "Personal attack", "BanTH", "Banglish"),
    ("banth/abusive", "Abusive or violent language", "BanTH", "Banglish"),
    ("banhate/hate", "Hate speech", "BanHate", "Bangla"),
    ("elearning/sentiment", "Sentiment of course comments", "eLearning", "Bangla + Banglish"),
    ("sib200/sib_topic", "Topic of a sentence (7)", "SIB-200", "Bangla"),
]
UNSEEN = [
    ("sentnob/sentiment", "Sentiment of social-media comments", "SentNoB", "Bangla"),
    ("indicsentiment/sentiment", "Sentiment of product reviews", "IndicSentiment", "Bangla"),
    ("indicxnli/entailment", "Does sentence A imply B? (yes/no)", "IndicXNLI", "Bangla"),
]
TRAINED_SOURCES = ("massive", "banglapsg", "banth", "banhate", "elearning", "sib200")


def metric(summary: dict, key: str, lang: str) -> dict:
    if key == "massive/intent (end-to-end)":
        return summary["massive_end_to_end_intent"][lang]
    return summary["tasks"][key][lang]["all"]


def pooled_accuracy(summary: dict, sources) -> float:
    """Accuracy over every test decision of the given sources, both question languages."""
    correct = total = 0
    for name, langs in summary["tasks"].items():
        if name.split("/")[0] in sources:
            for scopes in langs.values():
                correct += scopes["all"]["accuracy"] * scopes["all"]["n"]
                total += scopes["all"]["n"]
    return correct / total


def benchmark_table(base: dict, tuned: dict, rows) -> str:
    """A Markdown accuracy table: Bangla-worded / English-worded question for each model.

    The better model on each row, by mean accuracy over both wordings, is shown in bold.
    """
    lines = [
        "| Task | Dataset | Text | Test size | Base model | laya-bangla |",
        "| --- | --- | --- | ---: | :---: | :---: |",
    ]
    for key, task, dataset, script in rows:
        cells, means = [], []
        for summary in (base, tuned):
            acc = [metric(summary, key, lang)["accuracy"] for lang in ("bn", "en")]
            cells.append(f"{acc[0]:.0%} / {acc[1]:.0%}")
            means.append(sum(acc))
        best = 0 if means[0] > means[1] else 1
        cells[best] = f"**{cells[best]}**"
        n = metric(tuned, key, "bn")["n"]
        lines.append(f"| {task} | {dataset} | {script} | {n:,} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)
