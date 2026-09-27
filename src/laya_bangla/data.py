"""Load and validate small, versioned evaluation sets before model downloads."""

import json
import re
import tomllib
from pathlib import Path

SLICES = ("bangla", "banglish", "mixed")


def label_space(question: dict) -> list[str]:
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


def load_questions(path: Path) -> dict:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"{path}: expected a nonempty question object")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("choice", "noul", "score"):
            raise ValueError(f"{path}: {qid}: unsupported question type")
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            raise ValueError(f"{path}: {qid}: instructions must be nonempty text")
        criteria = q.get("criteria")
        if q["type"] == "choice":
            valid = isinstance(criteria, dict) and len(criteria) >= 2
        elif q["type"] == "score":
            valid = isinstance(criteria, list) and len(criteria) >= 2
        else:
            valid = isinstance(criteria, dict) and set(criteria) == {"false", "true"}
        if not valid:
            raise ValueError(f"{path}: {qid}: invalid {q['type']} criteria")
        descriptions = criteria.values() if isinstance(criteria, dict) else criteria
        if any(not isinstance(value, str) or not value.strip() for value in descriptions):
            raise ValueError(f"{path}: {qid}: criteria descriptions must be nonempty text")
    return questions


def load_examples(path: Path, questions: dict) -> list[dict]:
    examples, ids, groups = [], set(), {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        location = f"{path}:{line_number}"
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{location}: invalid JSON: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{location}: expected an object")
        for key in ("id", "group_id", "state"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"{location}: {key} must be nonempty text")
        if item["id"] in ids:
            raise ValueError(f"{location}: duplicate id {item['id']}")
        ids.add(item["id"])
        if item.get("slice") not in SLICES:
            raise ValueError(f"{location}: slice must be one of {SLICES}")
        expected = item.get("expected")
        if not isinstance(expected, dict) or set(expected) != set(questions):
            raise ValueError(f"{location}: expected labels must match question IDs exactly")
        for qid, q in questions.items():
            label = expected[qid]
            if q["type"] == "noul":
                valid = type(label) is bool
            elif q["type"] == "score":
                valid = type(label) is int and 0 <= label < len(q["criteria"])
            else:
                valid = isinstance(label, str) and label in q["criteria"]
            if not valid:
                raise ValueError(f"{location}: invalid expected label for {qid}: {label!r}")
        previous = groups.setdefault(item["group_id"], expected)
        if previous != expected:
            raise ValueError(f"{location}: translations in a group must have identical labels")
        examples.append(item)
    if not examples:
        raise ValueError(f"{path}: evaluation set is empty")
    return examples


def load_inputs(config_path: Path) -> tuple[dict, dict, list[dict]]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", config.get("revision", "")):
        raise ValueError("revision must be a full 40-character immutable Hub commit SHA")
    if not isinstance(config.get("model_id"), str) or not config["model_id"].strip():
        raise ValueError("model_id must be a nonempty Hub repository ID")
    for key in ("max_len", "head_max_len", "cpu_threads", "ece_bins"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "warmup_calls"):
        if type(config.get(key)) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["head_max_len"] + 4 >= config["max_len"]:
        raise ValueError("max_len must leave room for state tokens after head_max_len")
    if config.get("device") not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps, or cuda")
    for key in ("dataset", "questions_bn", "questions_en"):
        config[key] = str((config_path.parent / config[key]).resolve())
    schemas = {lang: load_questions(Path(config[f"questions_{lang}"])) for lang in ("bn", "en")}
    signatures = [
        [(qid, q["type"], label_space(q)) for qid, q in questions.items()]
        for questions in schemas.values()
    ]
    if signatures[0] != signatures[1]:
        raise ValueError(
            "Bangla and English schemas must share question IDs, types, and label order"
        )
    return config, schemas, load_examples(Path(config["dataset"]), schemas["bn"])
