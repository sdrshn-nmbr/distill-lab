import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import modal
from pydantic import TypeAdapter

from distill_lab.canonical import canonical_json
from distill_lab.checkpoint_identity import checkpoint_digest
from distill_lab.contracts import CandidateTokenMethod, FreshTraining, ResolvedRun, ResumeTraining
from distill_lab.miles_adapter import (
    launch_miles_training,
    verify_miles_checkout,
)
from distill_lab.planning import load_study, require_clean_harness, resolve_study
from distill_lab.receipts import AttemptRecorder, failure_code
from distill_lab.security import reject_credentials
from distill_lab.validation import ResumeEvidence, RunState

REMOTE_REPO = Path("/workspace/distill-lab")
REMOTE_MILES = Path("/workspace/miles")
MODEL_PATH = Path("/root/models/Qwen3.5-4B")
RESULT_ROOT = Path("/root/distill-lab-results")
_OBJECT = TypeAdapter(dict[str, Any])


def project_roots(*, is_local: bool, module_path: Path) -> tuple[Path, Path]:
    if not is_local:
        return REMOTE_REPO, REMOTE_MILES
    repository = module_path.resolve().parents[2]
    return repository, repository.parent / "miles"


def training_log_path(run_dir: Path, attempt_id: str) -> Path:
    return run_dir / f"train-{attempt_id}.log"


def ordered_training_logs(run_dir: Path) -> tuple[Path, ...]:
    logs: list[tuple[int, Path]] = []
    starts: set[int] = set()
    for evidence_path in run_dir.glob("evidence-*.json"):
        evidence = _OBJECT.validate_python(json.loads(evidence_path.read_text()))
        resumed_from = evidence.get("resumed_from_updates")
        if resumed_from is None:
            start = 0
        elif isinstance(resumed_from, int) and not isinstance(resumed_from, bool):
            start = resumed_from
        else:
            raise ValueError("training evidence has an invalid resume start")
        if start in starts:
            raise ValueError("duplicate resume start in training evidence")
        starts.add(start)
        attempt_id = evidence_path.stem.removeprefix("evidence-")
        log_path = training_log_path(run_dir, attempt_id)
        if not log_path.is_file():
            raise ValueError("training evidence has no matching log")
        logs.append((start, log_path))
    if not logs:
        raise ValueError("training run has no completed evidence")
    return tuple(path for _, path in sorted(logs))


def verify_candidate_checkpoint(run: ResolvedRun, training_data: str, model_path: Path) -> None:
    if not isinstance(run.source.method, CandidateTokenMethod):
        return
    rows = [
        _OBJECT.validate_python(json.loads(line))
        for line in training_data.splitlines()
        if line.strip()
    ]
    expected: set[str] = set()
    for row in rows:
        metadata = cast(object, row.get("metadata"))
        if not isinstance(metadata, dict):
            raise ValueError("candidate training data must name one checkpoint digest")
        typed_metadata = cast(dict[str, object], metadata)
        value = typed_metadata.get("checkpoint_sha256")
        if not isinstance(value, str):
            raise ValueError("candidate training data must name one checkpoint digest")
        expected.add(value)
    if len(expected) != 1:
        raise ValueError("candidate training data must name one checkpoint digest")
    if checkpoint_digest(model_path) not in expected:
        raise ValueError("candidate state checkpoint does not match the training model")


LOCAL_REPO, LOCAL_MILES = project_roots(
    is_local=modal.is_local(),
    module_path=Path(__file__),
)

_fixture = load_study(LOCAL_REPO / "experiments/fixtures/minimal.json")
_miles = _fixture.miles
_marker = json.dumps(
    {"revision": _miles.revision, "source_sha256": _miles.source_sha256},
    separators=(",", ":"),
)

app = modal.App("distill-lab-qwen35")
model_volume = modal.Volume.from_name("miles-opsd-models")
result_volume = modal.Volume.from_name("distill-lab-results", create_if_missing=True)

image_factory = cast(Any, modal.Image)
image = (
    image_factory.from_registry(_miles.image)
    .entrypoint([])
    .add_local_dir(
        LOCAL_REPO,
        remote_path=str(REMOTE_REPO),
        ignore=[".git", ".venv", "artifacts", ".pytest_cache", ".ruff_cache", "__pycache__"],
        copy=True,
    )
    .add_local_dir(
        LOCAL_MILES,
        remote_path=str(REMOTE_MILES),
        ignore=[".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__"],
        copy=True,
    )
    .run_commands(
        f"printf '%s' '{_marker}' > {REMOTE_MILES}/.distill-lab-source.json",
    )
    .env(
        {
            "PYTHONPATH": (
                f"{REMOTE_REPO}/src:{REMOTE_MILES}:/root/Megatron-LM:/sgl-workspace/sglang/python"
            ),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "NCCL_NVLS_ENABLE": "0",
        }
    )
)


modal_function: Any = cast(Any, app).function


@modal_function(
    image=image,
    gpu="H200:1",
    cpu=16,
    memory=131_072,
    timeout=7_500,
    volumes={"/root/models": model_volume, str(RESULT_ROOT): result_volume},
)
def train(
    resolved_run_json: str,
    training_data: str,
    training_data_sha256: str,
    run_tag: str,
    stop_after_updates: int | None = None,
    resume_completed_updates: int | None = None,
) -> dict[str, Any]:
    run = ResolvedRun.model_validate_json(resolved_run_json)
    _verify_packaged_harness(run)
    verify_miles_checkout(run, REMOTE_MILES)
    if hashlib.sha256(training_data.encode()).hexdigest() != training_data_sha256:
        raise ValueError("training data digest mismatch")
    verify_candidate_checkpoint(run, training_data, MODEL_PATH)
    run_dir = RESULT_ROOT / run.run_id / run_tag
    save_path = run_dir / "checkpoints"
    data_path = run_dir / "training.jsonl"
    recorder = AttemptRecorder(run=run, operation="miles_train", root=RESULT_ROOT)
    log_path = training_log_path(run_dir, recorder.attempt_id)
    gradient_path = save_path / "evidence" / recorder.attempt_id
    if resume_completed_updates is None:
        launch = FreshTraining(kind="fresh", stop_after_updates=stop_after_updates)
    else:
        marker = save_path / "latest_checkpointed_iteration.txt"
        launch = ResumeTraining(
            kind="resume",
            checkpoint_root=str(save_path),
            latest_marker_sha256=_file_digest(marker),
            completed_updates=resume_completed_updates,
        )
    if isinstance(launch, FreshTraining):
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("fresh Modal run directory is not empty")
        run_dir.mkdir(parents=True, exist_ok=True)
        data_path.write_text(training_data)
    elif data_path.read_text() != training_data:
        raise ValueError("resume training data does not match the first launch")
    telemetry_path = run_dir / f"gpu-{recorder.attempt_id}.jsonl"
    stop_telemetry = threading.Event()
    telemetry = threading.Thread(
        target=_sample_gpu,
        args=(telemetry_path, stop_telemetry, run.source.observability.gpu_sample_interval_seconds),
        daemon=True,
    )
    started = time.monotonic()
    ray_started = False
    try:
        write_system_snapshot(run_dir / f"system-before-{recorder.attempt_id}.json")
        subprocess.run(
            ["ray", "start", "--head", "--num-gpus=1", "--disable-usage-stats"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        ray_started = True
        telemetry.start()
        process = launch_miles_training(
            run=run,
            miles_checkout=REMOTE_MILES,
            model_path=MODEL_PATH,
            training_data=data_path,
            save_path=save_path,
            evidence_path=gradient_path,
            log_path=log_path,
            environment=dict(os.environ),
            launch=launch,
        )
    except BaseException as error:
        cleanup_runtime(
            telemetry=telemetry,
            stop_telemetry=stop_telemetry,
            ray_started=ray_started,
            snapshot_path=run_dir / f"system-after-{recorder.attempt_id}.json",
        )
        failure_path = run_dir / f"failure-{recorder.attempt_id}.json"
        write_private_failure(failure_path, error)
        recorder.fail(
            failure_code=failure_code(error),
            artifacts={"failure_evidence": _file_digest(failure_path)},
        )
        result_volume.commit()
        raise RuntimeError("Modal training attempt failed; inspect its private artifacts") from None
    try:
        cleanup_failures = cleanup_runtime(
            telemetry=telemetry,
            stop_telemetry=stop_telemetry,
            ray_started=ray_started,
            snapshot_path=run_dir / f"system-after-{recorder.attempt_id}.json",
        )
        if cleanup_failures:
            raise RuntimeError("runtime cleanup failed")
        evidence = _training_evidence(run_dir, save_path, gradient_path, log_path)
        evidence["elapsed_seconds"] = time.monotonic() - started
        evidence["return_code"] = process.returncode
        evidence["resumed_from_updates"] = resume_completed_updates
        evidence_path = run_dir / f"evidence-{recorder.attempt_id}.json"
        evidence_path.write_bytes(canonical_json(evidence))
    except BaseException as error:
        failure_path = run_dir / f"failure-{recorder.attempt_id}.json"
        write_private_failure(failure_path, error)
        recorder.fail(
            failure_code=failure_code(error),
            artifacts={"failure_evidence": _file_digest(failure_path)},
        )
        result_volume.commit()
        raise RuntimeError("Modal evidence gate failed; inspect its private artifacts") from None
    terminal = recorder.complete(
        artifacts={
            "training_evidence": _file_digest(evidence_path),
            "training_log": _file_digest(log_path),
            "gpu_telemetry": _file_digest(telemetry_path),
        }
    )
    result_volume.commit()
    return {"attempt": terminal.model_dump(mode="json"), "evidence": evidence}


@modal_function(
    image=image,
    gpu="H200:1",
    cpu=8,
    memory=65_536,
    timeout=1_800,
    volumes={"/root/models": model_volume},
)
def prepare_candidate_state(prompt: str) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(REMOTE_REPO / "src/distill_lab/modal_candidate_worker.py"),
            str(MODEL_PATH),
            prompt,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1_700,
    )
    return result.stdout.strip()


@modal_function(
    image=image,
    gpu="H200:1",
    cpu=16,
    memory=131_072,
    timeout=3_600,
    volumes={"/root/models": model_volume, str(RESULT_ROOT): result_volume},
)
def validate_phase_one(
    resolved_run_json: str,
    source_run_id: str,
    source_run_tag: str,
    iteration: int,
    training_row: str,
    miles_starting_loss: float,
    preflight_only: bool = False,
    padded_length: int | None = None,
    packing_patch: bool = False,
) -> dict[str, Any]:
    run = ResolvedRun.model_validate_json(resolved_run_json)
    _verify_packaged_harness(run)
    source = RESULT_ROOT / source_run_id / source_run_tag
    checkpoint = source / "checkpoints" / f"iter_{iteration:07d}"
    if not checkpoint.is_dir():
        raise ValueError("source checkpoint does not exist")
    row = _OBJECT.validate_python(json.loads(training_row))
    metadata = row.get("metadata")
    request: dict[str, Any] = {
        "base_path": str(MODEL_PATH),
        "checkpoint": str(checkpoint),
        "learning_rate": run.source.training.learning_rate,
        "miles_starting_loss": miles_starting_loss,
        "preflight_only": preflight_only,
        "padded_length": padded_length,
        "packing_patch": packing_patch,
    }
    if isinstance(run.source.method, CandidateTokenMethod):
        if not isinstance(metadata, dict):
            raise ValueError("candidate row metadata is missing")
        request.update(
            token_ids=metadata["token_ids"],
            response_length=metadata["response_length"],
        )
    else:
        request["messages"] = row["messages"]
    validation_dir = RESULT_ROOT / run.run_id / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    request_path = validation_dir / f"phase-one-{run.source.method.kind}-request.json"
    result_path = validation_dir / f"phase-one-{run.source.method.kind}.json"
    request_path.write_bytes(canonical_json(request))
    process = subprocess.run(
        [
            sys.executable,
            str(REMOTE_REPO / "src/distill_lab/modal_validation_worker.py"),
            str(request_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3_500,
    )
    if process.returncode != 0:
        failure_path = validation_dir / f"phase-one-{run.source.method.kind}-failure.json"
        write_private_failure(
            failure_path,
            RuntimeError(process.stderr[-16_384:] or "validation worker exited without stderr"),
        )
        result_volume.commit()
        raise RuntimeError("phase-one validation failed; inspect its private artifact")
    output_lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("phase-one validation worker returned no result")
    result = _OBJECT.validate_python(json.loads(output_lines[-1]))
    result_path.write_bytes(canonical_json(result))
    result_volume.commit()
    return result


@modal_function(
    image=image,
    gpu="H200:1",
    cpu=16,
    memory=131_072,
    timeout=5_400,
    volumes={"/root/models": model_volume, str(RESULT_ROOT): result_volume},
)
def validate_resume(
    resolved_run_json: str,
    continuous_tag: str,
    resumed_tag: str,
) -> dict[str, Any]:
    run = ResolvedRun.model_validate_json(resolved_run_json)
    _verify_packaged_harness(run)
    if run.source.budget.training_updates != 3:
        raise ValueError("resume validation requires exactly three updates")
    root = RESULT_ROOT / run.run_id
    continuous_dir = root / continuous_tag
    resumed_dir = root / resumed_tag
    continuous_data = continuous_dir / "training.jsonl"
    resumed_data = resumed_dir / "training.jsonl"
    if continuous_data.read_bytes() != resumed_data.read_bytes():
        raise ValueError("continuous and resumed runs used different training data")
    rows = [line for line in continuous_data.read_text().splitlines() if line.strip()]
    if len(rows) != 3:
        raise ValueError("resume validation requires three training rows")
    fixed_row = _OBJECT.validate_python(json.loads(rows[0]))
    raw_messages = fixed_row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("resume validation requires SFT messages")
    messages = cast(list[object], raw_messages)

    validation_dir = root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    states = {
        "continuous": _validate_resume_state(
            run_dir=continuous_dir,
            messages=messages,
            validation_dir=validation_dir,
            name="continuous",
        ),
        "resumed": _validate_resume_state(
            run_dir=resumed_dir,
            messages=messages,
            validation_dir=validation_dir,
            name="resumed",
        ),
    }
    evidence = ResumeEvidence(
        loss_tolerance=1e-6,
        continuous=states["continuous"],
        resumed=states["resumed"],
    )
    result_path = validation_dir / "resume-equivalence.json"
    result_path.write_bytes(canonical_json(evidence.model_dump(mode="json")))
    result_volume.commit()
    return evidence.model_dump(mode="json")


def _validate_resume_state(
    *,
    run_dir: Path,
    messages: list[object],
    validation_dir: Path,
    name: str,
) -> RunState:
    checkpoint = run_dir / "checkpoints" / "iter_0000003"
    dataset_state = run_dir / "checkpoints" / "rollout/global_dataset_state_dict_3.pt"
    if not checkpoint.is_dir() or not dataset_state.is_file():
        raise ValueError(f"{name} run has no final checkpoint or dataset state")
    request = {
        "operation": "resume_state",
        "base_path": str(MODEL_PATH),
        "checkpoint": str(checkpoint),
        "dataset_state": str(dataset_state),
        "log_paths": [str(path) for path in ordered_training_logs(run_dir)],
        "messages": messages,
        "padded_length": 128,
    }
    request_path = validation_dir / f"resume-{name}-request.json"
    request_path.write_bytes(canonical_json(request))
    process = subprocess.run(
        [
            sys.executable,
            str(REMOTE_REPO / "src/distill_lab/modal_validation_worker.py"),
            str(request_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=2_600,
    )
    if process.returncode != 0:
        failure_path = validation_dir / f"resume-{name}-failure.json"
        write_private_failure(
            failure_path,
            RuntimeError(process.stderr[-16_384:] or "resume worker exited without stderr"),
        )
        result_volume.commit()
        raise RuntimeError(f"{name} resume validation failed; inspect its private artifact")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{name} resume validation returned no result")
    state = RunState.model_validate_json(lines[-1])
    (validation_dir / f"resume-{name}.json").write_bytes(
        canonical_json(state.model_dump(mode="json"))
    )
    return state


@app.local_entrypoint()
def main(
    study: str = "experiments/pinapple-sft.json",
    training_data: str = "datasets/fixtures/pinapple_teacher_sft.jsonl",
    run_tag: str = "pinapple-sft-two-update",
    stop_after_updates: int | None = None,
    resume_completed_updates: int | None = None,
    candidate_state_out: str | None = None,
    phase_one_source_run: str | None = None,
    phase_one_source_tag: str | None = None,
    phase_one_iteration: int = 1,
    phase_one_starting_loss: float | None = None,
    phase_one_preflight_only: bool = False,
    phase_one_padded_length: int | None = None,
    phase_one_packing_patch: bool = False,
    resume_continuous_tag: str | None = None,
    resume_interrupted_tag: str | None = None,
) -> None:
    if candidate_state_out is not None:
        state = prepare_candidate_state.remote(
            "What fruit should I add to tomato soup? Answer in one sentence."
        )
        destination = LOCAL_REPO / candidate_state_out
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(state + "\n")
        print(state)
        return
    study_path = LOCAL_REPO / study
    run = resolve_study(load_study(study_path))
    require_clean_harness(run)
    verify_miles_checkout(run, LOCAL_MILES)
    if run.source.budget.training_updates < 1:
        raise ValueError("Modal training requires at least one update")
    data = (LOCAL_REPO / training_data).read_text()
    if resume_continuous_tag is not None or resume_interrupted_tag is not None:
        if resume_continuous_tag is None or resume_interrupted_tag is None:
            raise ValueError("resume validation requires both run tags")
        result = validate_resume.remote(
            run.model_dump_json(),
            resume_continuous_tag,
            resume_interrupted_tag,
        )
        print(json.dumps(result, indent=2))
        return
    if phase_one_source_run is not None:
        if phase_one_source_tag is None or phase_one_starting_loss is None:
            raise ValueError("phase-one validation requires source tag and starting loss")
        rows = [line for line in data.splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError("phase-one validation requires exactly one training row")
        result = validate_phase_one.remote(
            run.model_dump_json(),
            phase_one_source_run,
            phase_one_source_tag,
            phase_one_iteration,
            rows[0],
            phase_one_starting_loss,
            phase_one_preflight_only,
            phase_one_padded_length,
            phase_one_packing_patch,
        )
        print(json.dumps(result, indent=2))
        return
    result = train.remote(
        run.model_dump_json(),
        data,
        hashlib.sha256(data.encode()).hexdigest(),
        run_tag,
        stop_after_updates,
        resume_completed_updates,
    )
    print(json.dumps(result, indent=2))


def _verify_packaged_harness(run: ResolvedRun) -> None:
    source = hashlib.sha256()
    for path in sorted((REMOTE_REPO / "src/distill_lab").glob("*.py")):
        source.update(path.name.encode())
        source.update(b"\0")
        source.update(path.read_bytes())
        source.update(b"\0")
    if source.hexdigest() != run.harness.source_sha256:
        raise ValueError("packaged distill-lab source digest mismatch")
    checks = {
        REMOTE_REPO / "uv.lock": run.harness.lock_sha256,
        REMOTE_REPO / "schemas/study.schema.json": run.harness.study_schema_sha256,
        REMOTE_REPO / "src/distill_lab/codex_app_server.py": (
            run.harness.prompt_implementation_sha256
        ),
    }
    if any(_file_digest(path) != expected for path, expected in checks.items()):
        raise ValueError("packaged distill-lab identity mismatch")


def _training_evidence(
    run_dir: Path,
    save_path: Path,
    gradient_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    grad_paths = sorted(gradient_path.glob("*.pt"))
    if not grad_paths:
        raise RuntimeError("Miles wrote no gradient evidence")
    script = (
        "import json,sys,torch; "
        "print(json.dumps([float(torch.load(p, weights_only=False)) for p in sys.argv[1:]]))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", script, *map(str, grad_paths)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    grad_norms = json.loads(loaded.stdout)
    if not all(
        isinstance(value, float) and math.isfinite(value) and value > 0 for value in grad_norms
    ):
        raise RuntimeError("training did not produce finite nonzero gradients")
    marker = save_path / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise RuntimeError("Miles wrote no checkpoint marker")
    checkpoint_dirs = sorted(path.name for path in save_path.glob("iter_*") if path.is_dir())
    if not checkpoint_dirs:
        raise RuntimeError("Miles wrote no checkpoint directory")
    fsdp_forward = gradient_path / "fsdp-forward.jsonl"
    if not fsdp_forward.is_file():
        raise RuntimeError("Miles wrote no FSDP forward evidence")
    return {
        "checkpoint_directories": checkpoint_dirs,
        "checkpoint_marker_sha256": _file_digest(marker),
        "gradient_norms": grad_norms,
        "fsdp_forward_sha256": _file_digest(fsdp_forward),
        "log_sha256": _file_digest(log_path),
        "run_directory": str(run_dir),
    }


def _sample_gpu(path: Path, stop: threading.Event, interval: float) -> None:
    query = "timestamp,index,memory.used,memory.total,utilization.gpu,power.draw"
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        record = {
            "monotonic_seconds": time.monotonic(),
            "return_code": result.returncode,
            "samples": result.stdout.strip().splitlines(),
        }
        with path.open("a") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        stop.wait(interval)


def write_system_snapshot(path: Path) -> None:
    commands = {
        "processes": ["ps", "-eo", "pid,ppid,stat,etime,command"],
        "network": ["ss", "-lntp"],
        "gpu": ["nvidia-smi", "-q"],
    }
    values: dict[str, Any] = {"timestamp": time.time()}
    for name, command in commands.items():
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            values[name] = {"return_code": result.returncode, "output": result.stdout}
        except FileNotFoundError:
            values[name] = {"error": "command_unavailable"}
        except subprocess.TimeoutExpired:
            values[name] = {"error": "command_timed_out"}
    path.write_bytes(canonical_json(values))


def _stop_telemetry(telemetry: threading.Thread, stop: threading.Event) -> None:
    stop.set()
    if telemetry.is_alive():
        telemetry.join(timeout=10)
    if telemetry.is_alive():
        raise RuntimeError("GPU telemetry thread did not stop")


def _stop_ray(started: bool) -> None:
    if not started:
        return
    result = subprocess.run(
        ["ray", "stop", "--force"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Ray did not stop cleanly")


def cleanup_runtime(
    *,
    telemetry: Any,
    stop_telemetry: Any,
    ray_started: bool,
    snapshot_path: Path,
) -> list[str]:
    failures: list[str] = []
    actions = (
        ("telemetry_cleanup_failed", lambda: _stop_telemetry(telemetry, stop_telemetry)),
        ("ray_cleanup_failed", lambda: _stop_ray(ray_started)),
        ("system_snapshot_failed", lambda: write_system_snapshot(snapshot_path)),
    )
    for code, action in actions:
        try:
            action()
        except BaseException:
            failures.append(code)
    return failures


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_private_failure(path: Path, error: BaseException) -> None:
    evidence = canonical_json(
        {
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )
    try:
        reject_credentials(evidence.decode())
    except ValueError:
        evidence = canonical_json({"error_type": type(error).__name__, "message": "redacted"})
    path.write_bytes(evidence)
