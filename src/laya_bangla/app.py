"""A local web app that compares the fine-tuned checkpoint with the upstream baseline."""

import threading
import time
from pathlib import Path

from .data import load_questions
from .prepare import load_tasks, render_question

STATIC = Path(__file__).parent / "static"
PRESETS = {
    "support": ("department", "churn_risk", "urgency"),
    "grievance": ("grievance_category", "grievance_severity"),
    "moderation": ("hate", "personal_attack", "abusive"),
    "sentiment": ("sentiment",),
    "assistant": ("massive_scenario",),
    "topic": ("sib_topic",),
}


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


def route_intent(agent, tasks: dict, state: str, scenario: str, lang: str) -> dict | None:
    """Second routing stage: the intent within the predicted MASSIVE scenario."""
    task = tasks["massive_intent"]
    options = task["groups"].get(scenario)
    if not options:
        return None
    question = render_question(task, task["instructions"]["eval"][lang], lang, options)
    return agent.predict(state, {"massive_intent": question})["answers"]["massive_intent"]


def create_app(models: dict, descriptions: dict, tasks_path: Path, baseline_config: Path):
    """`models` maps a display name to a loaded Laya agent; `descriptions` says what each is."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field

    from .data import load_inputs

    tasks = load_tasks(tasks_path)
    config, _schemas, _examples = load_inputs(baseline_config)
    support = {lang: load_questions(Path(config[f"questions_{lang}"])) for lang in ("bn", "en")}
    lock = threading.Lock()  # Laya agents are not documented as thread-safe.
    app = FastAPI(title="laya-bangla", docs_url="/api/docs")

    class Request(BaseModel):
        text: str = Field(min_length=1, max_length=4000)
        preset: str = "support"
        lang: str = Field(default="bn", pattern="^(bn|en)$")
        questions: dict | None = None

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/info")
    def info():
        return {"models": descriptions, "presets": list(PRESETS) + ["custom"]}

    @app.get("/api/presets/{preset}")
    def preset(preset: str, lang: str = "bn"):
        try:
            return preset_questions(tasks, support, preset, lang)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/predict")
    def predict(request: Request):
        try:
            if request.preset == "custom":
                if not request.questions:
                    raise ValueError("custom requests need a questions object")
                questions = request.questions
            else:
                questions = preset_questions(tasks, support, request.preset, request.lang)
        except (KeyError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        results = {}
        for name, agent in models.items():
            start = time.perf_counter()
            try:
                with lock:
                    answers = agent.predict(request.text, questions)["answers"]
                    if request.preset == "assistant":
                        scenario = answers["massive_scenario"]["choice"]
                        intent = route_intent(agent, tasks, request.text, scenario, request.lang)
                        if intent:
                            answers["massive_intent"] = intent
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, f"{name}: {exc}") from exc
            results[name] = {
                "answers": answers,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        return {"questions": questions, "results": results}

    return app


def serve(
    model: Path,
    tasks_path: Path,
    baseline_config: Path,
    device: str,
    host: str,
    port: int,
    compare_base: bool,
) -> None:
    import laya
    import uvicorn

    from .benchmark import checkpoint_path
    from .cli import choose_device

    device = choose_device(device)
    models = {"fine-tuned": laya.load(str(model), device=device)}
    descriptions = {"fine-tuned": str(model)}
    if compare_base:
        path, source = checkpoint_path("base", baseline_config)
        models["base"] = laya.load(path, device=device)
        descriptions["base"] = f"{source['model_id']}@{source['revision'][:12]}"
    app = create_app(models, descriptions, tasks_path, baseline_config)
    print(f"Open http://{host}:{port}/ ({', '.join(models)} on {device})", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
