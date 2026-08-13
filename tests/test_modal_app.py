import json
import subprocess
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from distill_lab import modal_app
from distill_lab.planning import load_study, resolve_study


def test_project_roots_use_packaged_paths_in_modal() -> None:
    repository, miles = modal_app.project_roots(
        is_local=False,
        module_path=Path("/root/modal_app.py"),
    )

    assert repository == Path("/workspace/distill-lab")
    assert miles == Path("/workspace/miles")


def test_training_logs_are_attempt_scoped(tmp_path: Path) -> None:
    assert modal_app.training_log_path(tmp_path, "first") != modal_app.training_log_path(
        tmp_path, "resume"
    )


def test_candidate_training_requires_the_state_checkpoint(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    run = resolve_study(load_study(Path("experiments/fixtures/candidate.json")))
    data = json.dumps({"metadata": {"checkpoint_sha256": "a" * 64}})

    def matching_digest(_path: Path) -> str:
        return "a" * 64

    def different_digest(_path: Path) -> str:
        return "b" * 64

    monkeypatch.setattr(modal_app, "checkpoint_digest", matching_digest)

    modal_app.verify_candidate_checkpoint(run, data, tmp_path)

    monkeypatch.setattr(modal_app, "checkpoint_digest", different_digest)
    with pytest.raises(ValueError, match="does not match"):
        modal_app.verify_candidate_checkpoint(run, data, tmp_path)

    with pytest.raises(ValueError, match="must name one"):
        modal_app.verify_candidate_checkpoint(run, "{}", tmp_path)


def test_private_failure_records_diagnostic_without_credentials(tmp_path: Path) -> None:
    destination = tmp_path / "failure.json"

    modal_app.write_private_failure(destination, RuntimeError("dataset rejected"))

    assert json.loads(destination.read_text()) == {
        "error_type": "RuntimeError",
        "message": "dataset rejected",
    }


def test_private_failure_redacts_credential_shaped_messages(tmp_path: Path) -> None:
    destination = tmp_path / "failure.json"

    modal_app.write_private_failure(destination, RuntimeError("tskey-auth-" + "A" * 40))

    assert json.loads(destination.read_text()) == {
        "error_type": "RuntimeError",
        "message": "redacted",
    }


def test_system_snapshot_records_unavailable_commands(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    def run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "ss":
            raise FileNotFoundError("ss")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(modal_app.subprocess, "run", run)
    destination = tmp_path / "system.json"

    modal_app.write_system_snapshot(destination)

    snapshot = json.loads(destination.read_text())
    assert snapshot["processes"] == {"return_code": 0, "output": "ok"}
    assert snapshot["network"] == {"error": "command_unavailable"}
    assert snapshot["gpu"] == {"return_code": 0, "output": "ok"}


def test_runtime_cleanup_attempts_every_action(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fail_telemetry(*_args: object) -> None:
        calls.append("telemetry")
        raise RuntimeError("telemetry stuck")

    def fail_ray(_started: bool) -> None:
        calls.append("ray")
        raise RuntimeError("ray stuck")

    def snapshot(_path: Path) -> None:
        calls.append("snapshot")

    monkeypatch.setattr(modal_app, "_stop_telemetry", fail_telemetry)
    monkeypatch.setattr(modal_app, "_stop_ray", fail_ray)
    monkeypatch.setattr(modal_app, "write_system_snapshot", snapshot)

    failures = modal_app.cleanup_runtime(
        telemetry=object(),
        stop_telemetry=object(),
        ray_started=True,
        snapshot_path=tmp_path / "after.json",
    )

    assert calls == ["telemetry", "ray", "snapshot"]
    assert failures == ["telemetry_cleanup_failed", "ray_cleanup_failed"]


def test_runtime_cleanup_reports_snapshot_failure(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    def stop_telemetry(*_args: object) -> None:
        return None

    def stop_ray(_started: bool) -> None:
        return None

    monkeypatch.setattr(modal_app, "_stop_telemetry", stop_telemetry)
    monkeypatch.setattr(modal_app, "_stop_ray", stop_ray)

    def fail_snapshot(_path: Path) -> None:
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(modal_app, "write_system_snapshot", fail_snapshot)

    failures = modal_app.cleanup_runtime(
        telemetry=object(),
        stop_telemetry=object(),
        ray_started=True,
        snapshot_path=tmp_path / "after.json",
    )

    assert failures == ["system_snapshot_failed"]
