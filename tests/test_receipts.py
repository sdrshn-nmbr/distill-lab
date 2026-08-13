import json
from pathlib import Path

import pytest

from distill_lab.planning import load_study, resolve_study
from distill_lab.receipts import AttemptRecorder


def test_attempt_has_exactly_one_terminal_receipt(tmp_path: Path) -> None:
    run = resolve_study(load_study(Path("experiments/fixtures/minimal.json")))
    recorder = AttemptRecorder(run=run, operation="generate", root=tmp_path)

    completed = recorder.complete(artifacts={"manifest": "a" * 64})

    stored = json.loads(
        (tmp_path / run.run_id / "attempts" / f"{completed.attempt_id}.json").read_text()
    )
    assert stored["status"] == "completed"
    assert stored["harness_source_sha256"] == run.harness.source_sha256
    with pytest.raises(RuntimeError, match="terminal"):
        recorder.fail(failure_code="late_failure")


def test_failed_attempt_contains_code_not_exception_text(tmp_path: Path) -> None:
    run = resolve_study(load_study(Path("experiments/fixtures/minimal.json")))
    recorder = AttemptRecorder(run=run, operation="generate", root=tmp_path)

    receipt = recorder.fail(failure_code="teacher_unavailable")

    assert receipt.status == "failed"
    assert receipt.failure_code == "teacher_unavailable"
