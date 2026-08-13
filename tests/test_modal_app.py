from pathlib import Path

from pytest import MonkeyPatch

from distill_lab import modal_app


def test_project_roots_use_packaged_paths_in_modal() -> None:
    repository, miles = modal_app.project_roots(
        is_local=False,
        module_path=Path("/root/modal_app.py"),
    )

    assert repository == Path("/workspace/distill-lab")
    assert miles == Path("/workspace/miles")


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
    monkeypatch.setattr(modal_app, "_write_system_snapshot", snapshot)

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

    monkeypatch.setattr(modal_app, "_write_system_snapshot", fail_snapshot)

    failures = modal_app.cleanup_runtime(
        telemetry=object(),
        stop_telemetry=object(),
        ray_started=True,
        snapshot_path=tmp_path / "after.json",
    )

    assert failures == ["system_snapshot_failed"]
