"""Publish a checkpoint with its model card and benchmark summaries, and the playground Space."""

import re
import shutil
import tempfile
from pathlib import Path

from .prepare import sha256_file

# What `laya.load` downloads from a Hub repository, plus this project's provenance files.
CHECKPOINT_FILES = ("model.safetensors", "rl_agent_config.json", "tasks.json", "train_report.json")
CHECKPOINT_DIRS = ("encoder", "tokenizer")
REQUIRED = (*CHECKPOINT_FILES, "encoder/config.json", "tokenizer/tokenizer.json")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def stage_model(
    model: Path, card: Path, benchmarks: dict[str, Path], notices: list[Path], staging: Path
) -> list[str]:
    """Copy exactly the files to publish into `staging` and return their repository paths.

    Benchmark predictions quote dataset text, so only the metrics-only `summary.json` of each
    run is published; the datasets themselves are never redistributed.
    """
    for name in REQUIRED:
        if not (model / name).is_file():
            raise FileNotFoundError(f"{model} is missing {name}")
    text = card.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError(f"{card} must start with a YAML front-matter block")
    copies = {"README.md": card}
    copies.update({name: model / name for name in CHECKPOINT_FILES})
    for folder in CHECKPOINT_DIRS:
        for path in sorted((model / folder).rglob("*")):
            if path.is_file():
                copies[path.relative_to(model).as_posix()] = path
    for path in notices:
        copies[path.name] = path
    for name, run in benchmarks.items():
        if not NAME.match(name):
            raise ValueError(f"Invalid benchmark name {name!r}")
        if not (run / "summary.json").is_file():
            raise FileNotFoundError(f"{run} has no summary.json")
        copies[f"benchmark/{name}/summary.json"] = run / "summary.json"
    for target, source in copies.items():
        (staging / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, staging / target)
    return sorted(copies)


def verify_upload(api, repo_id: str, revision: str, staging: Path, files: list[str]) -> None:
    """Every staged file exists at `revision`, and every LFS file has the local SHA-256."""
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    for name in files:
        if name not in remote:
            raise RuntimeError(f"{name} is missing from {repo_id}@{revision}")
        lfs = remote[name].lfs
        if lfs is not None and lfs.sha256 != sha256_file(staging / name):
            raise RuntimeError(f"{name} on the Hub does not match the local file")


def publish_model(
    model: Path,
    repo_id: str,
    card: Path,
    benchmarks: dict[str, Path],
    notices: list[Path],
    tag: str | None,
    message: str,
    api=None,
) -> str:
    """Upload a checkpoint, verify it, tag the commit, and return the commit SHA."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    # Tags name released checkpoints; never move one to a different commit.
    if tag and tag in {ref.name for ref in api.list_repo_refs(repo_id).tags}:
        raise ValueError(f"{repo_id} already has a tag {tag!r}")
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        files = stage_model(model, card, benchmarks, notices, staging)
        commit = api.upload_folder(repo_id=repo_id, folder_path=staging, commit_message=message)
        verify_upload(api, repo_id, commit.oid, staging, files)
    if tag:
        api.create_tag(repo_id, tag=tag, revision=commit.oid)
    return commit.oid


def publish_space(space: Path, repo_id: str, message: str, api=None) -> str:
    """Upload the Gradio playground and return the Space commit SHA."""
    if not (space / "app.py").is_file() or not (space / "README.md").is_file():
        raise FileNotFoundError(f"{space} needs app.py and README.md")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=space,
        commit_message=message,
        ignore_patterns=["__pycache__/**", "*.pyc"],
    )
    return commit.oid
