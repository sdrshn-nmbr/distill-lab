import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from distill_lab.artifacts import LocalArtifactStore
from distill_lab.canonical import canonical_json
from distill_lab.contracts import (
    ArtifactRef,
    CandidateTokenMethod,
    FreshTraining,
    ResolvedRun,
    ResumeTraining,
    TrainingLaunch,
)

_OBJECT = TypeAdapter(dict[str, Any])
_SOURCE_ROOT = Path(__file__).resolve().parents[1]


def materialize_sft_training_data(
    *, store: LocalArtifactStore, manifest: ArtifactRef, output: Path
) -> Path:
    value = _manifest(store, manifest, "complete_response")
    rows: list[dict[str, Any]] = []
    for record in value["records"]:
        if not record["accepted"]:
            continue
        raw = _OBJECT.validate_python(
            json.loads(store.read_bytes(ArtifactRef.model_validate(record["raw_artifact"])))
        )
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": raw["example"]["prompt"]},
                    {"role": "assistant", "content": raw["teacher_output"]["text"]},
                ],
                "metadata": {
                    "example_id": record["example_id"],
                    "source_manifest": manifest.sha256,
                },
            }
        )
    return _write_rows(rows, output)


def materialize_candidate_training_data(
    *, store: LocalArtifactStore, manifest: ArtifactRef, output: Path
) -> Path:
    value = _manifest(store, manifest, "candidate_token")
    rows: list[dict[str, Any]] = []
    for record in value["records"]:
        if not record["accepted"]:
            continue
        raw = _OBJECT.validate_python(
            json.loads(store.read_bytes(ArtifactRef.model_validate(record["raw_artifact"])))
        )
        request = raw["request"]
        selected = raw["teacher_output"]["selected_token_id"]
        token_ids = request["prompt_token_ids"] + request["student_token_ids"] + [selected]
        response_length = len(request["student_token_ids"]) + 1
        rows.append(
            {
                "text": request["prompt"],
                "metadata": {
                    "state_id": record["state_id"],
                    "source_manifest": manifest.sha256,
                    "checkpoint_sha256": request["checkpoint_sha256"],
                    "token_ids": token_ids,
                    "response_length": response_length,
                    "loss_mask": [0] * (response_length - 1) + [1],
                },
            }
        )
    return _write_rows(rows, output)


def build_miles_command(
    *,
    run: ResolvedRun,
    miles_checkout: Path,
    model_path: Path,
    training_data: Path,
    save_path: Path,
    launch: TrainingLaunch | None = None,
) -> tuple[str, ...]:
    training = run.source.training
    method = run.source.method
    launch = launch or FreshTraining(kind="fresh")
    rollout_function = (
        "distill_lab.miles_rollout.generate_exact_token_rollout"
        if isinstance(method, CandidateTokenMethod)
        else "miles.rollout.sft_rollout.generate_rollout"
    )
    arguments = (
        "--hf-checkpoint",
        str(model_path),
        "--save",
        str(save_path),
        "--save-interval",
        str(training.checkpoint_interval),
        "--rollout-function-path",
        rollout_function,
        "--prompt-data",
        str(training_data),
        "--input-key",
        "messages" if not isinstance(method, CandidateTokenMethod) else "text",
        "--rollout-shuffle",
        "--num-rollout",
        str(run.source.budget.training_updates),
        "--rollout-batch-size",
        str(training.global_batch_size),
        "--global-batch-size",
        str(training.global_batch_size),
        "--loss-type",
        "sft_loss",
        "--calculate-per-token-loss",
        "--disable-compute-advantages-and-returns",
        "--debug-train-only",
        "--optimizer",
        "adam",
        "--lr",
        str(training.learning_rate),
        "--lr-decay-style",
        "constant",
        "--train-backend",
        "fsdp",
        "--actor-num-nodes",
        "1",
        "--actor-num-gpus-per-node",
        "1",
        "--gradient-checkpointing",
        "--attn-implementation",
        "flash_attention_3",
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu",
        str(training.max_sequence_tokens * training.micro_batch_size),
        "--ci-save-grad-norm",
        str(save_path / "evidence" / "{role}-{rollout_id}-{step_id}.pt"),
    )
    if not isinstance(method, CandidateTokenMethod):
        arguments += (
            "--apply-chat-template",
            "--apply-chat-template-kwargs",
            json.dumps({"enable_thinking": run.source.student.thinking_mode}),
        )
    else:
        arguments += ("--rollout-global-dataset",)
    if isinstance(launch, ResumeTraining):
        if launch.completed_updates >= run.source.budget.training_updates:
            raise ValueError("resume checkpoint already reached the training update budget")
        arguments += ("--load", launch.checkpoint_root)
    elif launch.stop_after_updates is not None:
        if launch.stop_after_updates >= run.source.budget.training_updates:
            raise ValueError("debug stop must leave at least one update for resume")
        arguments += ("--debug-exit-after-rollout", str(launch.stop_after_updates))
    return (*run.components.miles_command, str(miles_checkout / "train_async.py"), *arguments)


def verify_miles_checkout(run: ResolvedRun, checkout: Path) -> None:
    if not (checkout / "train_async.py").is_file():
        raise ValueError(f"Miles checkout is missing train_async.py: {checkout}")
    marker = checkout / ".distill-lab-source.json"
    if not (checkout / ".git").exists():
        if not marker.is_file():
            raise ValueError("packaged Miles checkout is missing its source marker")
        identity = _OBJECT.validate_python(json.loads(marker.read_text()))
        if identity != {
            "revision": run.source.miles.revision,
            "source_sha256": run.source.miles.source_sha256,
        }:
            raise ValueError("packaged Miles source identity mismatch")
        return
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if head != run.source.miles.revision:
        raise ValueError(
            f"Miles revision mismatch: expected {run.source.miles.revision}, got {head}"
        )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    if dirty:
        raise ValueError("Miles checkout must be clean")
    if git_tree_digest(checkout) != run.source.miles.source_sha256:
        raise ValueError("Miles tracked source digest mismatch")


def git_tree_digest(checkout: Path) -> str:
    paths = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=checkout,
        check=True,
        capture_output=True,
        timeout=10,
    ).stdout.split(b"\0")
    digest = hashlib.sha256()
    for raw_path in paths:
        if not raw_path:
            continue
        relative = raw_path.decode()
        path = checkout / relative
        digest.update(raw_path)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<gitlink>")
        digest.update(b"\0")
    return digest.hexdigest()


def training_child_environment(source: dict[str, str], *, isolated_home: Path) -> dict[str, str]:
    allowed = {
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_DEBUG",
        "NCCL_NVLS_ENABLE",
        "PATH",
        "PYTHONPATH",
        "TMPDIR",
        "USER",
    }
    environment = {name: source[name] for name in sorted(allowed & source.keys())}
    python_paths = [str(_SOURCE_ROOT)]
    if existing := environment.get("PYTHONPATH"):
        python_paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    isolated_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    isolated_home.chmod(0o700)
    environment["HOME"] = str(isolated_home)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def launch_miles_training(
    *,
    run: ResolvedRun,
    miles_checkout: Path,
    model_path: Path,
    training_data: Path,
    save_path: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
    launch: TrainingLaunch | None = None,
) -> subprocess.CompletedProcess[str]:
    verify_miles_checkout(run, miles_checkout)
    if not training_data.is_file():
        raise ValueError(f"training data does not exist: {training_data}")
    launch = launch or FreshTraining(kind="fresh")
    if isinstance(launch, FreshTraining) and save_path.exists() and any(save_path.iterdir()):
        raise ValueError(f"training output directory is not empty: {save_path}")
    if isinstance(launch, ResumeTraining):
        if Path(launch.checkpoint_root).resolve() != save_path.resolve():
            raise ValueError("resume checkpoint root must equal the training save path")
        marker = save_path / "latest_checkpointed_iteration.txt"
        if not marker.is_file():
            raise ValueError("resume checkpoint is missing its latest-iteration marker")
        if hashlib.sha256(marker.read_bytes()).hexdigest() != launch.latest_marker_sha256:
            raise ValueError("resume checkpoint marker digest mismatch")
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "evidence").mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_miles_command(
        run=run,
        miles_checkout=miles_checkout,
        model_path=model_path,
        training_data=training_data,
        save_path=save_path,
        launch=launch,
    )
    with log_path.open("x") as log:
        return subprocess.run(
            command,
            cwd=miles_checkout,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=run.source.training.timeout_seconds,
            env=training_child_environment(
                environment or dict(os.environ), isolated_home=save_path / ".home"
            ),
        )


def _manifest(store: LocalArtifactStore, reference: ArtifactRef, kind: str) -> dict[str, Any]:
    value = _OBJECT.validate_python(json.loads(store.read_bytes(reference)))
    if value.get("method", {}).get("kind") != kind:
        raise ValueError(f"expected {kind} manifest")
    return value


def _write_rows(rows: list[dict[str, Any]], output: Path) -> Path:
    if not rows:
        raise ValueError("training dataset has no accepted rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json(row) for row in rows)
    output.write_bytes(payload)
    return output
