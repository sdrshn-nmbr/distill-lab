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

from distill_lab.canonical import canonical_json
from distill_lab.contracts import FreshTraining, ResolvedRun, ResumeTraining
from distill_lab.miles_adapter import (
    launch_miles_training,
    verify_miles_checkout,
)
from distill_lab.planning import load_study, require_clean_harness, resolve_study
from distill_lab.receipts import AttemptRecorder, failure_code

REMOTE_REPO = Path("/workspace/distill-lab")
REMOTE_MILES = Path("/workspace/miles")
MODEL_PATH = Path("/root/models/Qwen3.5-4B")
RESULT_ROOT = Path("/root/distill-lab-results")


def project_roots(*, is_local: bool, module_path: Path) -> tuple[Path, Path]:
    if not is_local:
        return REMOTE_REPO, REMOTE_MILES
    repository = module_path.resolve().parents[2]
    return repository, repository.parent / "miles"


def training_log_path(run_dir: Path, attempt_id: str) -> Path:
    return run_dir / f"train-{attempt_id}.log"


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
        recorder.fail(failure_code=failure_code(error))
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
        recorder.fail(failure_code=failure_code(error))
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


@app.local_entrypoint()
def main(
    study: str = "experiments/pinapple-sft.json",
    training_data: str = "datasets/fixtures/pinapple_teacher_sft.jsonl",
    run_tag: str = "pinapple-sft-two-update",
    stop_after_updates: int | None = None,
    resume_completed_updates: int | None = None,
) -> None:
    study_path = LOCAL_REPO / study
    run = resolve_study(load_study(study_path))
    require_clean_harness(run)
    verify_miles_checkout(run, LOCAL_MILES)
    if run.source.budget.training_updates < 1:
        raise ValueError("Modal training requires at least one update")
    data = (LOCAL_REPO / training_data).read_text()
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
    return {
        "checkpoint_directories": checkpoint_dirs,
        "checkpoint_marker_sha256": _file_digest(marker),
        "gradient_norms": grad_norms,
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
